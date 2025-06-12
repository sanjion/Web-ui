# Web UI - 使用uv进行依赖管理

这个项目已完全迁移到使用uv作为Python包管理器，在当前目录下使用.venv虚拟环境。

## 快速开始

### 1. 安装uv (如果尚未安装)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 创建环境并安装依赖

```bash
# 创建虚拟环境并安装所有依赖
uv sync --no-install-project
```

### 3. 运行项目

```bash
# 使用uv运行（推荐）
uv run python webui.py

# 带参数运行
uv run python webui.py --ip 0.0.0.0 --port 7788 --theme Ocean
```

## 日常使用

### 运行项目

```bash
# 最简单的运行方式
uv run python webui.py

# 或者激活环境后运行
source .venv/bin/activate
python webui.py
```

### 管理依赖

#### 添加新依赖

```bash
# 添加运行时依赖
uv add package-name

# 添加开发依赖
uv add --dev package-name
```

#### 更新依赖

```bash
uv lock --upgrade
uv sync --no-install-project
```

#### 查看已安装的包

```bash
uv tree
```

## 开发工具

项目包含了一些可选的开发工具：

```bash
# 代码格式化
uv run black .

# 代码检查
uv run flake8 .

# 类型检查
uv run mypy .

# 运行测试
uv run pytest
```

## 环境变量

在项目根目录创建`.env`文件配置API密钥：

```env
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
MISTRAL_API_KEY=your_mistral_api_key
GOOGLE_API_KEY=your_google_api_key
```

## 故障排除

### 重新创建环境

如果遇到环境问题：

```bash
# 删除现有环境
rm -rf .venv

# 重新创建并安装依赖
uv sync --no-install-project
```

### 清理缓存

```bash
uv cache clean
```

## 文件说明

- `pyproject.toml` - 项目配置和依赖声明
- `uv.lock` - 锁定的依赖版本（类似package-lock.json）
- `.venv/` - 虚拟环境目录
- `webui.py` - 主程序入口
