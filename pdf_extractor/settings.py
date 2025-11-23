# -*- coding: utf-8 -*-
import json
import os

CONFIG_FILE = "config.json"

def get_default_settings():
    """返回默认设置字典"""
    return {
        "mode": 3,
        "dpi": 300,
        "parallel": True,
        "optimize": False,
        "output_format": "png",
        "target_numbers": "",
        "extract_tables": True,
        "imgsz": 1024,
        "conf": 0.4,
        "table_min_conf": 0.25,
        "padding": 10,
        "output_dir": "output",
        "window_width": 450,
        "window_height": 600
    }

def load_settings():
    """从 config.json 加载设置，如果文件不存在则返回默认设置"""
    if not os.path.exists(CONFIG_FILE):
        return get_default_settings()
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            user_settings = json.load(f)
            # 合并，用保存的设置覆盖默认设置，以兼容未来新增的默认参数
            settings = get_default_settings()
            settings.update(user_settings)
            return settings
    except (json.JSONDecodeError, TypeError):
        # 如果文件损坏或格式不正确，返回默认设置
        return get_default_settings()

def save_settings(settings_dict):
    """将当前设置保存到 config.json"""
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(settings_dict, f, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"错误: 无法保存设置文件: {e}")

