import argparse
import asyncio
import glob
import logging
import os
import re
import traceback

import gradio as gr
from browser_use.agent.service import Agent
from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import (
    BrowserContextWindowSize,
)
from dotenv import load_dotenv
from gradio.themes import Base, Citrus, Default, Glass, Monochrome, Ocean, Origin, Soft

from src.agent.custom_agent import CustomAgent
from src.agent.custom_prompts import CustomAgentMessagePrompt, CustomSystemPrompt
from src.browser.custom_browser import CustomBrowser
from src.browser.custom_context import BrowserContextConfig
from src.controller.custom_controller import CustomController
from src.utils import utils
from src.utils.agent_state import AgentState
from src.utils.default_config_settings import (
    default_config,
    save_current_config,
    update_ui_from_config,
)
from src.utils.utils import (
    MissingAPIKeyError,
    capture_screenshot,
    get_latest_files,
    update_model_dropdown,
)

load_dotenv()
logger = logging.getLogger(__name__)

# Global variables for persistence
_global_browser = None
_global_browser_context = None
_global_agent = None
_global_agent_state = AgentState()

# Theme configuration
THEME_MAP = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean(),
    "Base": Base(),
}

CSS_STYLES = """
.gradio-container {
    max-width: 1200px !important;
    margin: auto !important;
    padding-top: 20px !important;
}
.header-text {
    text-align: center;
    margin-bottom: 30px;
}
.theme-section {
    margin-bottom: 20px;
    padding: 15px;
    border-radius: 10px;
}
"""


def resolve_sensitive_env_variables(text):
    """Replace environment variable placeholders ($SENSITIVE_*) with their values."""
    if not text:
        return text

    env_vars = re.findall(r"\$SENSITIVE_[A-Za-z0-9_]*", text)
    result = text
    for var in env_vars:
        env_name = var[1:]  # Remove the $ prefix
        env_value = os.getenv(env_name)
        if env_value is not None:
            result = result.replace(var, env_value)
    return result


async def stop_agent():
    """Request the agent to stop and update UI with feedback."""
    global _global_agent

    try:
        if _global_agent is not None:
            _global_agent.stop()

        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        return (
            gr.update(value="Stopping...", interactive=False),
            gr.update(interactive=False),
        )
    except Exception as e:
        logger.error(f"Error during stop: {str(e)}")
        return (gr.update(value="Stop", interactive=True), gr.update(interactive=True))


async def stop_research_agent():
    """Request the research agent to stop and update UI with feedback."""
    global _global_agent_state

    try:
        _global_agent_state.request_stop()
        message = "Stop requested - the agent will halt at the next safe point"
        logger.info(f"🛑 {message}")

        return (
            gr.update(value="Stopping...", interactive=False),
            gr.update(interactive=False),
        )
    except Exception as e:
        logger.error(f"Error during stop: {str(e)}")
        return (gr.update(value="Stop", interactive=True), gr.update(interactive=True))


def get_browser_config(
    use_own_browser, headless, disable_security, window_w, window_h, chrome_cdp
):
    """Create browser configuration."""
    extra_chromium_args = [f"--window-size={window_w},{window_h}"]
    cdp_url = chrome_cdp
    chrome_path = None

    if use_own_browser:
        cdp_url = os.getenv("CHROME_CDP", chrome_cdp)
        chrome_path = os.getenv("CHROME_PATH", None)
        if chrome_path == "":
            chrome_path = None
        chrome_user_data = os.getenv("CHROME_USER_DATA", None)
        if chrome_user_data:
            extra_chromium_args += [f"--user-data-dir={chrome_user_data}"]

    return BrowserConfig(
        headless=headless,
        cdp_url=cdp_url,
        disable_security=disable_security,
        chrome_instance_path=chrome_path,
        extra_chromium_args=extra_chromium_args,
    ), cdp_url


def get_context_config(save_trace_path, save_recording_path, window_w, window_h):
    """Create browser context configuration."""
    return BrowserContextConfig(
        trace_path=save_trace_path if save_trace_path else None,
        save_recording_path=save_recording_path if save_recording_path else None,
        no_viewport=False,
        browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
    )


async def setup_browser_and_context(
    browser_config, context_config, agent_type, cdp_url
):
    """Setup browser and context with proper error handling."""
    global _global_browser, _global_browser_context

    browser_class = CustomBrowser if agent_type == "custom" else Browser

    if (_global_browser is None) or (cdp_url and cdp_url != "" and cdp_url is not None):
        _global_browser = browser_class(config=browser_config)

    if _global_browser_context is None or (
        cdp_url and cdp_url != "" and cdp_url is not None
    ):
        _global_browser_context = await _global_browser.new_context(
            config=context_config
        )

    return _global_browser, _global_browser_context


async def cleanup_browser_resources(keep_browser_open):
    """Cleanup browser resources based on configuration."""
    global _global_browser, _global_browser_context, _global_agent

    _global_agent = None

    if not keep_browser_open:
        if _global_browser_context:
            await _global_browser_context.close()
            _global_browser_context = None

        if _global_browser:
            await _global_browser.close()
            _global_browser = None


async def run_agent_common(agent_type, llm, browser_config, context_config, **kwargs):
    """Common agent execution logic."""
    global _global_agent

    try:
        cdp_url = browser_config.cdp_url
        browser, browser_context = await setup_browser_and_context(
            browser_config, context_config, agent_type, cdp_url
        )

        if agent_type == "org":
            _global_agent = Agent(
                task=kwargs["task"],
                llm=llm,
                use_vision=kwargs["use_vision"],
                browser=browser,
                browser_context=browser_context,
                max_actions_per_step=kwargs["max_actions_per_step"],
                tool_calling_method=kwargs["tool_calling_method"],
                max_input_tokens=kwargs["max_input_tokens"],
                generate_gif=True,
            )
        else:  # custom
            controller = CustomController()
            _global_agent = CustomAgent(
                task=kwargs["task"],
                add_infos=kwargs.get("add_infos", ""),
                use_vision=kwargs["use_vision"],
                llm=llm,
                browser=browser,
                browser_context=browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                agent_prompt_class=CustomAgentMessagePrompt,
                max_actions_per_step=kwargs["max_actions_per_step"],
                tool_calling_method=kwargs["tool_calling_method"],
                max_input_tokens=kwargs["max_input_tokens"],
                generate_gif=True,
            )

        history = await _global_agent.run(max_steps=kwargs["max_steps"])

        # Save history
        history_file = os.path.join(
            kwargs["save_agent_history_path"], f"{_global_agent.state.agent_id}.json"
        )
        _global_agent.save_history(history_file)

        # Extract results
        final_result = history.final_result()
        errors = history.errors()
        model_actions = history.model_actions()
        model_thoughts = history.model_thoughts()
        trace_file = get_latest_files(kwargs["save_trace_path"])

        return (
            final_result,
            errors,
            model_actions,
            model_thoughts,
            trace_file.get(".zip"),
            history_file,
        )

    except Exception as e:
        traceback.print_exc()
        error_msg = f"{str(e)}\n{traceback.format_exc()}"
        return "", error_msg, "", "", None, None

    finally:
        await cleanup_browser_resources(kwargs["keep_browser_open"])


async def run_browser_agent(
    agent_type,
    llm_provider,
    llm_model_name,
    llm_num_ctx,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_own_browser,
    keep_browser_open,
    headless,
    disable_security,
    window_w,
    window_h,
    save_recording_path,
    save_agent_history_path,
    save_trace_path,
    enable_recording,
    task,
    add_infos,
    max_steps,
    use_vision,
    max_actions_per_step,
    tool_calling_method,
    chrome_cdp,
    max_input_tokens,
):
    """Main browser agent execution function."""
    try:
        # Disable recording if checkbox is unchecked
        if not enable_recording:
            save_recording_path = None

        # Ensure recording directory exists
        if save_recording_path:
            os.makedirs(save_recording_path, exist_ok=True)

        task = resolve_sensitive_env_variables(task)

        # Get LLM model
        llm = utils.get_llm_model(
            provider=llm_provider,
            model_name=llm_model_name,
            num_ctx=llm_num_ctx,
            temperature=llm_temperature,
            base_url=llm_base_url,
            api_key=llm_api_key,
        )

        # Get configurations
        browser_config, cdp_url = get_browser_config(
            use_own_browser, headless, disable_security, window_w, window_h, chrome_cdp
        )
        context_config = get_context_config(
            save_trace_path, save_recording_path, window_w, window_h
        )

        # Run agent
        result = await run_agent_common(
            agent_type=agent_type,
            llm=llm,
            browser_config=browser_config,
            context_config=context_config,
            task=task,
            add_infos=add_infos,
            max_steps=max_steps,
            use_vision=use_vision,
            max_actions_per_step=max_actions_per_step,
            tool_calling_method=tool_calling_method,
            max_input_tokens=max_input_tokens,
            save_agent_history_path=save_agent_history_path,
            save_trace_path=save_trace_path,
            keep_browser_open=keep_browser_open,
        )

        gif_path = os.path.join(os.path.dirname(__file__), "agent_history.gif")

        return (
            *result,
            gif_path,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True),
        )

    except MissingAPIKeyError as e:
        logger.error(str(e))
        raise gr.Error(str(e), print_exception=False) from e

    except Exception as e:
        traceback.print_exc()
        errors = f"{str(e)}\n{traceback.format_exc()}"
        return (
            "",
            errors,
            "",
            "",
            None,
            None,
            None,
            gr.update(value="Stop", interactive=True),
            gr.update(interactive=True),
        )


async def run_with_stream(*args):
    """Stream execution wrapper with live browser view."""
    global _global_agent

    # Extract arguments
    (
        agent_type,
        llm_provider,
        llm_model_name,
        llm_num_ctx,
        llm_temperature,
        llm_base_url,
        llm_api_key,
        use_own_browser,
        keep_browser_open,
        headless,
        disable_security,
        window_w,
        window_h,
        save_recording_path,
        save_agent_history_path,
        save_trace_path,
        enable_recording,
        task,
        add_infos,
        max_steps,
        use_vision,
        max_actions_per_step,
        tool_calling_method,
        chrome_cdp,
        max_input_tokens,
    ) = args

    stream_vw = 80
    stream_vh = int(80 * window_h // window_w)

    if not headless:
        result = await run_browser_agent(*args)
        yield [gr.update(visible=False)] + list(result)
    else:
        try:
            # Run agent in background
            agent_task = asyncio.create_task(run_browser_agent(*args))

            # Initialize streaming values
            html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Using browser...</h1>"
            empty_results = ["", "", "", "", None, None, None]

            # Stream updates while agent runs
            while not agent_task.done():
                try:
                    encoded_screenshot = await capture_screenshot(
                        _global_browser_context
                    )
                    if encoded_screenshot is not None:
                        html_content = f'<img src="data:image/jpeg;base64,{encoded_screenshot}" style="width:{stream_vw}vw; height:{stream_vh}vh; border:1px solid #ccc;">'
                    else:
                        html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"
                except Exception:
                    html_content = f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Waiting for browser session...</h1>"

                # Check if agent stopped
                if _global_agent and _global_agent.state.stopped:
                    yield [
                        gr.HTML(value=html_content, visible=True),
                        *empty_results,
                        gr.update(value="Stopping...", interactive=False),
                        gr.update(interactive=False),
                    ]
                    break
                else:
                    yield [
                        gr.HTML(value=html_content, visible=True),
                        *empty_results,
                        gr.update(),
                        gr.update(),
                    ]

                await asyncio.sleep(0.1)

            # Get final results
            try:
                result = await agent_task
                final_results = result[:-2]  # Exclude button updates
                button_updates = result[-2:]
            except gr.Error:
                final_results = empty_results
                button_updates = [
                    gr.update(value="Stop", interactive=True),
                    gr.update(interactive=True),
                ]
            except Exception as e:
                final_results = ["", f"Agent error: {str(e)}", "", "", None, None, None]
                button_updates = [
                    gr.update(value="Stop", interactive=True),
                    gr.update(interactive=True),
                ]

            yield [
                gr.HTML(value=html_content, visible=True),
                *final_results,
                *button_updates,
            ]

        except Exception as e:
            yield [
                gr.HTML(
                    value=f"<h1 style='width:{stream_vw}vw; height:{stream_vh}vh'>Error occurred</h1>",
                    visible=True,
                ),
                "",
                f"Error: {str(e)}\n{traceback.format_exc()}",
                "",
                "",
                None,
                None,
                None,
                gr.update(value="Stop", interactive=True),
                gr.update(interactive=True),
            ]


async def close_global_browser():
    """Close global browser resources."""
    global _global_browser, _global_browser_context

    if _global_browser_context:
        await _global_browser_context.close()
        _global_browser_context = None

    if _global_browser:
        await _global_browser.close()
        _global_browser = None


async def run_deep_search(
    research_task,
    max_search_iteration_input,
    max_query_per_iter_input,
    llm_provider,
    llm_model_name,
    llm_num_ctx,
    llm_temperature,
    llm_base_url,
    llm_api_key,
    use_vision,
    use_own_browser,
    headless,
    chrome_cdp,
):
    """Run deep research with the configured parameters."""
    from src.utils.deep_research import deep_research

    global _global_agent_state

    _global_agent_state.clear_stop()

    llm = utils.get_llm_model(
        provider=llm_provider,
        model_name=llm_model_name,
        num_ctx=llm_num_ctx,
        temperature=llm_temperature,
        base_url=llm_base_url,
        api_key=llm_api_key,
    )

    markdown_content, file_path = await deep_research(
        research_task,
        llm,
        _global_agent_state,
        max_search_iterations=max_search_iteration_input,
        max_query_num=max_query_per_iter_input,
        use_vision=use_vision,
        headless=headless,
        use_own_browser=use_own_browser,
        chrome_cdp=chrome_cdp,
    )

    return (
        markdown_content,
        file_path,
        gr.update(value="Stop", interactive=True),
        gr.update(interactive=True),
    )


def list_recordings(save_recording_path):
    """List all recordings in the specified path."""
    if not os.path.exists(save_recording_path):
        return []

    recordings = glob.glob(
        os.path.join(save_recording_path, "*.[mM][pP]4")
    ) + glob.glob(os.path.join(save_recording_path, "*.[wW][eE][bB][mM]"))

    recordings.sort(key=os.path.getctime)

    return [
        (recording, f"{idx}. {os.path.basename(recording)}")
        for idx, recording in enumerate(recordings, start=1)
    ]


def create_ui(config, theme_name="Ocean"):
    """Create the main UI interface."""
    with gr.Blocks(
        title="Browser Use WebUI", theme=THEME_MAP[theme_name], css=CSS_STYLES
    ) as demo:
        # Header
        gr.Markdown(
            "# 🌐 Browser Use WebUI\n### Control your browser with AI assistance",
            elem_classes=["header-text"],
        )

        with gr.Tabs():
            # LLM Settings Tab
            with gr.TabItem("🔧 LLM Settings", id=1), gr.Group():
                llm_provider = gr.Dropdown(
                    choices=list(utils.model_names.keys()),
                    label="LLM Provider",
                    value=config["llm_provider"],
                    info="Select your preferred language model provider",
                )

                llm_model_name = gr.Dropdown(
                    label="Model Name",
                    choices=utils.model_names["openai"],
                    value=config["llm_model_name"],
                    allow_custom_value=True,
                    info="Select a model or type a custom model name",
                )

                llm_num_ctx = gr.Slider(
                    minimum=2**8,
                    maximum=2**16,
                    value=config["llm_num_ctx"],
                    step=1,
                    label="Max Context Length",
                    info="Controls max context length (less = faster)",
                    visible=config["llm_provider"] == "ollama",
                )

                llm_temperature = gr.Slider(
                    minimum=0.0,
                    maximum=2.0,
                    value=config["llm_temperature"],
                    step=0.1,
                    label="Temperature",
                    info="Controls randomness in model outputs",
                )

                with gr.Row():
                    llm_base_url = gr.Textbox(
                        label="Base URL",
                        value=config["llm_base_url"],
                        info="API endpoint URL (if required)",
                    )
                    llm_api_key = gr.Textbox(
                        label="API Key",
                        type="password",
                        value=config["llm_api_key"],
                        info="Your API key (leave blank to use .env)",
                    )

            # Agent Settings Tab
            with gr.TabItem("⚙️ Agent Settings", id=2), gr.Group():
                agent_type = gr.Radio(
                    ["org", "custom"],
                    label="Agent Type",
                    value=config["agent_type"],
                    info="Select the type of agent to use",
                )

                with gr.Row():
                    max_steps = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=config["max_steps"],
                        step=1,
                        label="Max Run Steps",
                        info="Maximum number of steps the agent will take",
                    )
                    max_actions_per_step = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=config["max_actions_per_step"],
                        step=1,
                        label="Max Actions per Step",
                        info="Maximum number of actions the agent will take per step",
                    )

                with gr.Row():
                    use_vision = gr.Checkbox(
                        label="Use Vision",
                        value=config["use_vision"],
                        info="Enable visual processing capabilities",
                    )
                    max_input_tokens = gr.Number(
                        label="Max Input Tokens", value=128000, precision=0
                    )

                tool_calling_method = gr.Dropdown(
                    label="Tool Calling Method",
                    value=config["tool_calling_method"],
                    choices=["auto", "json_schema", "function_calling"],
                    allow_custom_value=True,
                    info="Tool Calls Function Name",
                    visible=False,
                )

            # Browser Settings Tab
            with gr.TabItem("🌐 Browser Settings", id=3), gr.Group():
                with gr.Row():
                    use_own_browser = gr.Checkbox(
                        label="Use Own Browser",
                        value=config["use_own_browser"],
                        info="Use your existing browser instance",
                    )
                    keep_browser_open = gr.Checkbox(
                        label="Keep Browser Open",
                        value=config["keep_browser_open"],
                        info="Keep Browser Open between Tasks",
                    )
                    headless = gr.Checkbox(
                        label="Headless Mode",
                        value=config["headless"],
                        info="Run browser without GUI",
                    )
                    disable_security = gr.Checkbox(
                        label="Disable Security",
                        value=config["disable_security"],
                        info="Disable browser security features",
                    )
                    enable_recording = gr.Checkbox(
                        label="Enable Recording",
                        value=config["enable_recording"],
                        info="Enable saving browser recordings",
                    )

                with gr.Row():
                    window_w = gr.Number(
                        label="Window Width",
                        value=config["window_w"],
                        info="Browser window width",
                    )
                    window_h = gr.Number(
                        label="Window Height",
                        value=config["window_h"],
                        info="Browser window height",
                    )

                chrome_cdp = gr.Textbox(
                    label="CDP URL",
                    placeholder="http://localhost:9222",
                    value="",
                    info="CDP for google remote debugging",
                )

                save_recording_path = gr.Textbox(
                    label="Recording Path",
                    placeholder="e.g. ./tmp/record_videos",
                    value=config["save_recording_path"],
                    info="Path to save browser recordings",
                )

                save_trace_path = gr.Textbox(
                    label="Trace Path",
                    placeholder="e.g. ./tmp/traces",
                    value=config["save_trace_path"],
                    info="Path to save Agent traces",
                )

                save_agent_history_path = gr.Textbox(
                    label="Agent History Save Path",
                    placeholder="e.g., ./tmp/agent_history",
                    value=config["save_agent_history_path"],
                    info="Specify the directory where agent history should be saved.",
                )

            # Run Agent Tab
            with gr.TabItem("🤖 Run Agent", id=4):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value=config["task"],
                    info="Describe what you want the agent to do",
                )

                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task",
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

                browser_view = gr.HTML(
                    value="<h1 style='width:80vw; height:50vh'>Waiting for browser session...</h1>",
                    label="Live Browser View",
                    visible=False,
                )

                gr.Markdown("### Results")
                with gr.Row():
                    final_result_output = gr.Textbox(label="Final Result", lines=3)
                    errors_output = gr.Textbox(label="Errors", lines=3)

                with gr.Row():
                    model_actions_output = gr.Textbox(
                        label="Model Actions", lines=3, visible=False
                    )
                    model_thoughts_output = gr.Textbox(
                        label="Model Thoughts", lines=3, visible=False
                    )

                recording_gif = gr.Image(label="Result GIF", format="gif")
                trace_file = gr.File(label="Trace File")
                agent_history_file = gr.File(label="Agent History")

            # Recordings Tab
            with gr.TabItem("🎥 Recordings", id=5):
                recordings_gallery = gr.Gallery(
                    label="Recordings",
                    value=list_recordings(config["save_recording_path"]),
                    columns=3,
                    height="auto",
                    object_fit="contain",
                )
                refresh_button = gr.Button("🔄 Refresh Recordings", variant="secondary")

            # Deep Research Tab
            with gr.TabItem("🧐 Deep Research", id=6):
                research_task_input = gr.Textbox(
                    label="Research Task",
                    lines=5,
                    value="Compose a report on the use of Reinforcement Learning for training Large Language Models...",
                )

                with gr.Row():
                    max_search_iteration_input = gr.Number(
                        label="Max Search Iteration", value=3, precision=0
                    )
                    max_query_per_iter_input = gr.Number(
                        label="Max Query per Iteration", value=1, precision=0
                    )

                with gr.Row():
                    research_button = gr.Button(
                        "▶️ Run Deep Research", variant="primary", scale=2
                    )
                    stop_research_button = gr.Button("⏹ Stop", variant="stop", scale=1)

                markdown_output_display = gr.Markdown(label="Research Report")
                markdown_download = gr.File(label="Download Research Report")

            # Configuration Tab
            with gr.TabItem("📁 UI Configuration", id=7):
                config_file_input = gr.File(
                    label="Load UI Settings from Config File",
                    file_types=[".pkl"],
                )

                with gr.Row():
                    load_config_button = gr.Button("Load Config", variant="primary")
                    save_config_button = gr.Button(
                        "Save UI Settings", variant="primary"
                    )

                config_status = gr.Textbox(label="Status", lines=2, interactive=False)

        # Event handlers
        def update_llm_num_ctx_visibility(provider):
            return gr.update(visible=provider == "ollama")

        # Bind events
        llm_provider.change(update_llm_num_ctx_visibility, llm_provider, llm_num_ctx)
        llm_provider.change(
            lambda provider, api_key, base_url: update_model_dropdown(
                provider, api_key, base_url
            ),
            [llm_provider, llm_api_key, llm_base_url],
            llm_model_name,
        )

        enable_recording.change(
            lambda enabled: gr.update(interactive=enabled),
            enable_recording,
            save_recording_path,
        )

        use_own_browser.change(fn=close_global_browser)
        keep_browser_open.change(fn=close_global_browser)

        # Button clicks
        stop_button.click(stop_agent, outputs=[stop_button, run_button])
        stop_research_button.click(
            stop_research_agent, outputs=[stop_research_button, research_button]
        )

        run_button.click(
            run_with_stream,
            inputs=[
                agent_type,
                llm_provider,
                llm_model_name,
                llm_num_ctx,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                keep_browser_open,
                headless,
                disable_security,
                window_w,
                window_h,
                save_recording_path,
                save_agent_history_path,
                save_trace_path,
                enable_recording,
                task,
                add_infos,
                max_steps,
                use_vision,
                max_actions_per_step,
                tool_calling_method,
                chrome_cdp,
                max_input_tokens,
            ],
            outputs=[
                browser_view,
                final_result_output,
                errors_output,
                model_actions_output,
                model_thoughts_output,
                recording_gif,
                trace_file,
                agent_history_file,
                stop_button,
                run_button,
            ],
        )

        research_button.click(
            run_deep_search,
            inputs=[
                research_task_input,
                max_search_iteration_input,
                max_query_per_iter_input,
                llm_provider,
                llm_model_name,
                llm_num_ctx,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_vision,
                use_own_browser,
                headless,
                chrome_cdp,
            ],
            outputs=[
                markdown_output_display,
                markdown_download,
                stop_research_button,
                research_button,
            ],
        )

        refresh_button.click(list_recordings, save_recording_path, recordings_gallery)

        load_config_button.click(
            update_ui_from_config,
            config_file_input,
            [
                agent_type,
                max_steps,
                max_actions_per_step,
                use_vision,
                tool_calling_method,
                llm_provider,
                llm_model_name,
                llm_num_ctx,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                keep_browser_open,
                headless,
                disable_security,
                enable_recording,
                window_w,
                window_h,
                save_recording_path,
                save_trace_path,
                save_agent_history_path,
                task,
                config_status,
            ],
        )

        save_config_button.click(
            save_current_config,
            [
                agent_type,
                max_steps,
                max_actions_per_step,
                use_vision,
                tool_calling_method,
                llm_provider,
                llm_model_name,
                llm_num_ctx,
                llm_temperature,
                llm_base_url,
                llm_api_key,
                use_own_browser,
                keep_browser_open,
                headless,
                disable_security,
                enable_recording,
                window_w,
                window_h,
                save_recording_path,
                save_trace_path,
                save_agent_history_path,
                task,
            ],
            config_status,
        )

    return demo


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument(
        "--ip", type=str, default="127.0.0.1", help="IP address to bind to"
    )
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument(
        "--theme",
        type=str,
        default="Ocean",
        choices=THEME_MAP.keys(),
        help="Theme to use for the UI",
    )
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    config_dict = default_config()
    demo = create_ui(config_dict, theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)


if __name__ == "__main__":
    main()
