#!/bin/bash

# ✅ 强制要求第一个参数存在（任务名）
if [ -z "$1" ]; then
    echo "❌ 错误：必须传入任务名（如 cp / cir / compat）"
    echo "👉 用法：bash run_by_date.sh <task_name> [mode]"
    exit 1
fi

# 🎯 参数解析
TASK_NAME="$1"
MODE="${2:-train-valid}"  # 第二个参数可选，默认是 train-valid

# 📅 构造分支名（按当天日期）
DATE_STR=$(date +%F)
BRANCH_NAME="tangshaokun/$DATE_STR"
DATE_CN=$(date -d "$DATE_STR" +'%Y年%m月%d日')

echo "📅 当前日期：$DATE_CN"
echo "🌿 正在切换到远程分支：$BRANCH_NAME"

# 🔧 Git 操作
git fetch origin
git reset --hard origin/$BRANCH_NAME || {
    echo "❌ 切换风分支失败: origin/$BRANCH_NAME [分支不存在]"
    exit 1
}
echo "✅ 分支切换成功：$BRANCH_NAME"
# 🚀 启动训练任务
echo "🚀 正在运行任务：$TASK_NAME，模式：$MODE"
torchrun --standalone --nproc_per_node=4 ./src/trains/run/${TASK_NAME}.py --mode=${MODE}
