# -*- coding: utf-8 -*-
"""
PDF内容提取多功能工具 (V38.2 - 引入递归XY Cut排序算法)

功能概述:
- 模式1: 从PDF中直接提取嵌入的图片对象。
- 模式2: 将PDF的每一页渲染并保存为图片（支持并行处理）。
- 模式3: 利用ONNX AI模型（如YOLO）检测并裁剪PDF页面中的特定区域（如图片、表格）。

V38.2 更新日志:
- 引入 MinerU 的 Recursive XY Cut 算法，优化复杂版面（如双栏、跨栏）下的图片/表格排序逻辑。
- 修复了双栏文档中图片序号混乱的问题 (Column-Major sorting)。
"""

import argparse
import ast
import concurrent.futures
import io
import os
import time
from typing import Dict, List, Optional, Set, Tuple

import cv2
import fitz  # PyMuPDF
import numpy as np
import onnx
import onnxruntime
from PIL import Image

# ==============================================================================
# 算法模块: Recursive XY Cut (Reading Order Sorting)
# ==============================================================================

def projection_by_bboxes(boxes: np.ndarray, axis: int) -> np.ndarray:
    """
    Get projection profile by bounding boxes.
    Args:
        boxes: [N, 4] (x1, y1, x2, y2)
        axis: 0 for X-axis (vertical projection), 1 for Y-axis (horizontal projection)
    """
    assert axis in [0, 1]
    if boxes.size == 0:
        return np.zeros(0, dtype=int)
        
    length = np.max(boxes[:, axis::2])
    res = np.zeros(length, dtype=int)
    for start, end in boxes[:, axis::2]:
        start = max(0, int(start))
        end = min(length, int(end))
        if start < end:
            res[start:end] += 1
    return res

def split_projection_profile(arr_values: np.ndarray, min_value: float, min_gap: float):
    """
    Split projection profile.
    Returns (start_indices, end_indices) arrays.
    """
    arr_index = np.where(arr_values > min_value)[0]
    if not len(arr_index):
        return None

    arr_diff = arr_index[1:] - arr_index[0:-1]
    arr_diff_index = np.where(arr_diff > min_gap)[0]
    arr_zero_intvl_start = arr_index[arr_diff_index]
    arr_zero_intvl_end = arr_index[arr_diff_index + 1]

    arr_start = np.insert(arr_zero_intvl_end, 0, arr_index[0])
    arr_end = np.append(arr_zero_intvl_start, arr_index[-1])
    arr_end += 1

    return arr_start, arr_end

def recursive_xy_cut(boxes: np.ndarray, indices: List[int], res: List[int]):
    """
    Recursive XY Cut algorithm tailored for Document Layout Analysis.
    Prioritizes Vertical Cuts (Columns) over Horizontal Cuts (Rows) to handle dual-column layouts correctly.
    
    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        indices: Original indices of the boxes
        res: Result list to append sorted indices to
    """
    if len(boxes) == 0:
        return

    # 1. Try splitting by X-axis (Vertical Columns) first
    x_projection = projection_by_bboxes(boxes, axis=0)
    pos_x = split_projection_profile(x_projection, 0, 1)

    if pos_x is not None and len(pos_x[0]) > 1:
        arr_x0, arr_x1 = pos_x
        for c0, c1 in zip(arr_x0, arr_x1):
            # Filter boxes that fall significantly into this column range
            box_centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
            mask = (box_centers_x >= c0) & (box_centers_x < c1)
            
            if np.any(mask):
                recursive_xy_cut(boxes[mask], indices[mask], res)
        return

    # 2. If X-split failed (Single Column), try splitting by Y-axis (Rows)
    y_projection = projection_by_bboxes(boxes, axis=1)
    pos_y = split_projection_profile(y_projection, 0, 1)

    if pos_y is not None and len(pos_y[0]) > 1:
        arr_y0, arr_y1 = pos_y
        for r0, r1 in zip(arr_y0, arr_y1):
            box_centers_y = (boxes[:, 1] + boxes[:, 3]) / 2
            mask = (box_centers_y >= r0) & (box_centers_y < r1)
            
            if np.any(mask):
                recursive_xy_cut(boxes[mask], indices[mask], res)
        return

    # 3. If both splits failed, sort by Top-to-Bottom, then Left-to-Right
    combined = list(zip(indices, boxes))
    combined.sort(key=lambda x: (x[1][1], x[1][0]))
    
    sorted_indices = [x[0] for x in combined]
    res.extend(sorted_indices)


# ==============================================================================
# 模块 1 & 2: PDF原生图片提取 和 页面到图片的转换
# ==============================================================================

def parse_number_ranges(range_string: Optional[str]) -> Optional[Set[int]]:
    """
    解析逗号分隔的数字和范围字符串 (例如 '1,3,5-8')。
    """
    if not range_string:
        return None
    selected_numbers = set()
    parts = range_string.split(',')
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            if '-' in part:
                start, end = map(int, part.split('-'))
                selected_numbers.update(range(start, end + 1))
            else:
                selected_numbers.add(int(part))
        except ValueError:
            print(f"[警告] 无法解析序号 '{part}'，将予以忽略。")
    return selected_numbers


def extract_embedded_images(pdf_path: str, output_dir: str, target_numbers: Optional[Set[int]] = None):
    """
    模式1：直接从PDF文件中提取嵌入的原始图片对象。
    """
    print("[*] [模式1] 扫描PDF以提取嵌入的图片对象...")
    try:
        pdf_file = fitz.open(pdf_path)
    except Exception as e:
        print(f"[错误] 无法打开PDF文件: {e}")
        return

    all_potential_images = []
    for page_index in range(len(pdf_file)):
        for img_info in pdf_file.get_page_images(page_index, full=True):
            rects = pdf_file.get_page_image_rects(page_index, img_info)
            if rects:
                all_potential_images.append({"xref": img_info[0], "bbox": rects[0], "page": page_index})

    if not all_potential_images:
        print("\n[完成] 未在文档中找到任何可提取的图片对象。")
        pdf_file.close()
        return

    # 模式1依旧保留简单的排序逻辑，因为它是基于对象流的，不一定完全对应视觉布局
    # 但为了更准确，也可以按 (page, y, x) 排序
    all_potential_images.sort(key=lambda img: (img["page"], img["bbox"].y0, img["bbox"].x0))

    saved_counter = 0
    for idx, img_info in enumerate(all_potential_images, 1):
        if target_numbers and idx not in target_numbers:
            continue

        base_image = pdf_file.extract_image(img_info["xref"])
        if not base_image or "image" not in base_image:
            continue

        output_path = os.path.join(output_dir, f"fig{idx}.png")
        with open(output_path, "wb") as f:
            f.write(base_image["image"])
        saved_counter += 1
        print(f"[*] 已保存图片: {output_path}")

    pdf_file.close()
    print("\n--------------------")
    print(f"[完成] 成功提取并保存了 {saved_counter} 张图片。")


def _render_page_worker(page_index: int, pdf_path: str, output_dir: str, dpi: int, optimize: bool, out_format: str) -> str:
    """
    为并行处理设计的单个页面渲染工作函数。
    """
    try:
        doc = fitz.open(pdf_path)
        page = doc[page_index]
        pix = page.get_pixmap(dpi=dpi)
        doc.close()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        output_path = os.path.join(output_dir, f"page_{page_index + 1}.{out_format}")

        save_options = {}
        if optimize:
            if out_format == 'png':
                save_options = {'optimize': True, 'compress_level': 9}
            elif out_format == 'webp':
                save_options = {'quality': 95, 'method': 6}

        img.save(output_path, format=out_format.upper(), **save_options)
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        status = "已优化并保存" if optimize else "已保存页面"
        return f"{status}: {output_path} (大小: {size_mb:.2f} MB)"
    except Exception as e:
        return f"错误：处理页面 {page_index + 1} 失败 - {e}"


def convert_pages_to_images(pdf_path: str, output_dir: str, dpi: int, parallel: bool, optimize: bool, out_format: str):
    """
    模式2：将PDF页面转换为图片。
    """
    print(f"[*] [模式2] 开始将PDF页面转换为 {out_format.upper()}...")
    try:
        doc = fitz.open(pdf_path)
        num_pages = len(doc)
        doc.close()
    except Exception as e:
        print(f"[错误] 无法打开或读取PDF文件: {e}")
        return

    start_time = time.time()
    if parallel:
        num_cores = min(os.cpu_count() or 1, 8, num_pages)
        print(f"[*] 运行模式: 并行 (使用 {num_cores} 个核心)")
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
            futures = [executor.submit(_render_page_worker, i, pdf_path, output_dir, dpi, optimize, out_format) for i in range(num_pages)]
            for future in concurrent.futures.as_completed(futures):
                print(f"[*] {future.result()}")
    else:
        print("[*] 运行模式: 单线程")
        for i in range(num_pages):
            result = _render_page_worker(i, pdf_path, output_dir, dpi, optimize, out_format)
            print(f"[*] {result}")

    duration = time.time() - start_time
    print("\n--------------------")
    print(f"[完成] 成功转换并保存了 {num_pages} 个页面。")
    print(f"[*] 模式2总耗时: {duration:.2f} 秒。")


# ==============================================================================
# 模块 3: AI视觉识别与提取
# ==============================================================================

def _preprocess_image(image: np.ndarray, new_shape: Tuple[int, int], stride: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    对图像进行预处理以适配YOLO模型输入。
    """
    h, w = image.shape[:2]
    new_h, new_w = new_shape
    ratio = min(new_h / h, new_w / w)
    resized_h, resized_w = int(round(h * ratio)), int(round(w * ratio))
    
    if ratio != 1:
        image = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        
    pad_w = (new_w - resized_w) % stride
    pad_h = (new_h - resized_h) % stride
    top, bottom = pad_h // 2, pad_h - (pad_h // 2)
    left, right = pad_w // 2, pad_w - (pad_w // 2)
    
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
    
    image = image.astype(np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    return image[np.newaxis, ...], ratio, (left, top)


def _scale_boxes(img1_shape: Tuple[int, int], boxes: np.ndarray, img0_shape: Tuple[int, int], pad_info: Tuple[int, int]) -> np.ndarray:
    """
    _scale_boxes function implementation.
    """
    gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
    pad_x, pad_y = pad_info
    boxes[..., 0] = (boxes[..., 0] - pad_x) / gain
    boxes[..., 1] = (boxes[..., 1] - pad_y) / gain
    boxes[..., 2] = (boxes[..., 2] - pad_x) / gain
    boxes[..., 3] = (boxes[..., 3] - pad_y) / gain
    boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, img0_shape[1])
    boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, img0_shape[0])
    return boxes


def _calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Calculates the Intersection over Union (IoU) of two bounding boxes."""
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    if inter_area == 0:
        return 0.0

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def _non_max_suppression(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
    """Performs Non-Maximal Suppression to remove overlapping bounding boxes."""
    if len(boxes) == 0:
        return []

    sorted_indices = np.argsort(scores)[::-1]
    kept_indices = []
    while len(sorted_indices) > 0:
        current_index = sorted_indices[0]
        kept_indices.append(current_index)
        if len(sorted_indices) == 1:
            break
        
        current_box = boxes[current_index]
        remaining_indices = sorted_indices[1:]
        
        ious = np.array([_calculate_iou(current_box, boxes[i]) for i in remaining_indices])
        
        below_threshold_indices = np.where(ious < iou_threshold)[0]
        
        sorted_indices = remaining_indices[below_threshold_indices]
        
    return kept_indices


def _sort_detected_objects(objects: List[Dict]) -> List[Dict]:
    """
    Sorts detected objects based on Recursive XY Cut (Reading Order).
    """
    if not objects:
        return []

    # Prepare boxes for XY Cut: [x0, y0, x1, y1]
    boxes_list = []
    for obj in objects:
        b = obj['bbox']
        boxes_list.append([max(0, int(b.x0)), max(0, int(b.y0)), max(0, int(b.x1)), max(0, int(b.y1))])
    
    boxes_np = np.array(boxes_list, dtype=int)
    indices = np.arange(len(boxes_np))
    res_indices = []

    try:
        recursive_xy_cut(boxes_np, indices, res_indices)
    except Exception as e:
        print(f"[警告] XYCut 排序失败: {e}。将回退到基础排序。")
        return sorted(objects, key=lambda obj: (obj['bbox'].y0, obj['bbox'].x0))

    if len(res_indices) != len(objects):
        print(f"[警告] XYCut 返回的索引数量不匹配 ({len(res_indices)} vs {len(objects)})。将回退到基础排序。")
        return sorted(objects, key=lambda obj: (obj['bbox'].y0, obj['bbox'].x0))

    sorted_objects = [objects[i] for i in res_indices]
    return sorted_objects


def _save_images(doc: fitz.Document, objects: List[Dict], output_dir: str, prefix: str, dpi: int, padding: int, target_numbers: Optional[Set[int]], out_format: str, optimize: bool):
    """
    将检测到的对象区域裁剪并使用Pillow保存为图片，支持多种格式和优化。
    """
    if not objects:
        return 0
        
    print(f"\n[*] AI模型共检测到 {len(objects)} 个有效{prefix}区域，准备保存...")
    saved_count = 0
    for idx, obj in enumerate(objects, 1):
        if target_numbers and idx not in target_numbers:
            continue
            
        bbox = obj['bbox']
        if bbox.is_empty or bbox.width <= 10 or bbox.height <= 10:
            continue
            
        # 使用clip从页面安全地裁剪区域
        clip_rect = (bbox + (-padding, -padding, padding, padding)).irect
        pix = doc[obj['page']].get_pixmap(clip=clip_rect, dpi=dpi)
        
        output_path = os.path.join(output_dir, f"{prefix}{idx}.{out_format}")
        
        try:
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            save_options = {}
            if out_format == 'webp':
                save_options['quality'] = 95
                if optimize:
                    save_options['method'] = 6
            elif out_format == 'png' and optimize:
                save_options = {'optimize': True, 'compress_level': 9}

            img.save(output_path, format=out_format.upper(), **save_options)

            saved_count += 1
            print(f"[*] 已保存图片: {output_path}")
        except Exception as e:
            print(f"[警告] 无法保存图片 {prefix}{idx}: {e}")

    print(f"[完成] 成功提取并保存了 {saved_count} 张{prefix}。")
    return saved_count


def extract_images_ai_yolo(pdf_path: str, output_dir: str, model_path: str, **kwargs):
    """
    模式3：使用ONNX YOLO模型检测并提取PDF中的图片和表格。
    """
    try:
        model = onnx.load(model_path)
        metadata = {prop.key: prop.value for prop in model.metadata_props}
        stride = ast.literal_eval(metadata.get("stride", "32"))
        class_names = ast.literal_eval(metadata.get("names", "{{}}"))
        print(f"[*] 从模型元数据加载: stride={stride}, classes={class_names}")
    except Exception as e:
        print(f"[错误] 无法加载或解析ONNX模型 '{model_path}': {e}")
        return

    device = kwargs['device']
    providers = ['CUDAExecutionProvider'] if 'cuda' in device.lower() and 'CUDAExecutionProvider' in onnxruntime.get_available_providers() else ['CPUExecutionProvider']
    session = onnxruntime.InferenceSession(model_path, providers=providers)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    doc = fitz.open(pdf_path)
    print(f"[*] [模式3] 运行模式: {'GPU' if 'cuda' in providers[0].lower() else 'CPU'}")
    print(f"[*] 正在对 {len(doc)} 个页面进行AI推理...")

    TABLE_MIN_CONF = 0.25
    TABLE_MIN_ASPECT_RATIO, TABLE_MAX_ASPECT_RATIO, TABLE_MIN_AREA_RATIO = 0.3, 3.0, 0.01

    figures_by_page, tables_by_page = {}, {}
    
    for page_num, page in enumerate(doc):
        print(f"  - 正在处理第 {page_num + 1}/{len(doc)} 页...")
        
        pix = page.get_pixmap(dpi=150) # 推理使用较低DPI
        img_cv = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n), cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
        tensor, _, pad_info = _preprocess_image(img_cv, (kwargs['imgsz'], kwargs['imgsz']), stride)
        outputs = session.run([output_name], {input_name: tensor})[0]

        # --- NMS后处理 ---
        raw_detections = outputs[0]
        if len(raw_detections) == 0:
            continue

        boxes, scores, class_ids = [], [], []
        for det in raw_detections:
            boxes.append(det[:4])
            scores.append(det[4])
            class_ids.append(int(det[5]))

        boxes = np.array(boxes)
        scores = np.array(scores)
        class_ids = np.array(class_ids)

        kept_indices = []
        unique_class_ids = np.unique(class_ids)
        for class_id in unique_class_ids:
            class_mask = (class_ids == class_id)
            class_boxes = boxes[class_mask]
            class_scores = scores[class_mask]
            
            iou_threshold = 0.45 # 标准NMS阈值
            
            indices_for_class = _non_max_suppression(class_boxes, class_scores, iou_threshold)
            
            original_indices = np.where(class_mask)[0]
            kept_indices.extend(original_indices[indices_for_class])

        page_figs, page_tbls = [], []
        for i in kept_indices:
            det = raw_detections[i]
            confidence, class_id = det[4], int(det[5])
            
            cls_name = class_names.get(class_id, '').lower()
            is_figure = cls_name == 'figure'
            is_table = cls_name == 'table' and kwargs['extract_tables']

            if not (is_figure or is_table): continue

            passes_conf = confidence > kwargs['conf']
            if is_table and confidence > TABLE_MIN_CONF: passes_conf = True
            
            if not passes_conf: continue

            scaled_box = _scale_boxes((kwargs['imgsz'], kwargs['imgsz']), det[:4][np.newaxis, ...], img_cv.shape[:2], pad_info)[0]
            
            if is_table:
                box_w, box_h = scaled_box[2] - scaled_box[0], scaled_box[3] - scaled_box[1]
                if box_h <= 0 or box_w <= 0: continue
                aspect_ratio = box_w / box_h
                area_ratio = (box_w * box_h) / (img_cv.shape[0] * img_cv.shape[1])
                if not (TABLE_MIN_ASPECT_RATIO < aspect_ratio < TABLE_MAX_ASPECT_RATIO and area_ratio > TABLE_MIN_AREA_RATIO):
                    continue
            
            page_rect = page.rect
            bbox = fitz.Rect(
                scaled_box[0] * page_rect.width / img_cv.shape[1],
                scaled_box[1] * page_rect.height / img_cv.shape[0],
                scaled_box[2] * page_rect.width / img_cv.shape[1],
                scaled_box[3] * page_rect.height / img_cv.shape[0]
            )
            
            result = {'page': page_num, 'bbox': bbox, 'conf': confidence}
            if is_figure: page_figs.append(result)
            elif is_table: page_tbls.append(result)
        
        if page_figs: figures_by_page[page_num] = page_figs
        if page_tbls: tables_by_page[page_num] = page_tbls
    
    # 整合、排序和保存结果
    all_figures, all_tables = [], []
    for page_num in sorted(figures_by_page.keys()):
        all_figures.extend(_sort_detected_objects(figures_by_page[page_num]))
    if kwargs['extract_tables']:
        for page_num in sorted(tables_by_page.keys()):
            all_tables.extend(_sort_detected_objects(tables_by_page[page_num]))

    _save_images(doc, all_figures, output_dir, "fig", kwargs['dpi'], kwargs['padding'], kwargs['target_numbers'], kwargs.get('out_format', 'png'), kwargs.get('optimize', False))
    if kwargs['extract_tables']:
        _save_images(doc, all_tables, output_dir, "table", kwargs['dpi'], kwargs['padding'], None, kwargs.get('out_format', 'png'), kwargs.get('optimize', False))

    if not all_figures and not all_tables:
        print("\n[完成] AI模型未在此文档中检测到任何'figure'或'table'区域。")
        
    doc.close()

# ==============================================================================
# 主程序入口
# ==============================================================================
def main():
    """主函数：解析命令行参数并执行相应任务。"""
    parser = argparse.ArgumentParser(
        description="从PDF中提取内容的多功能工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # --- 通用参数 ---
    parser.add_argument("-i", "--input", required=True, help="输入的PDF文件路径。")
    parser.add_argument("-o", "--output", default="output", help="输出文件夹路径。(默认: 'output')")
    parser.add_argument("--mode", type=int, default=3, choices=[1, 2, 3], 
                        help="1: 提取嵌入对象\n" 
                             "2: 页面转图片\n" 
                             "3: AI视觉提取 (默认)")
    # --- 模式 2 & 3 参数 ---
    parser.add_argument("--dpi", type=int, default=300, help="[模式2/3] 裁剪或渲染的分辨率。(默认: 300)")
    parser.add_argument("--format", type=str, default='png', choices=['png', 'webp'], help="[模式2/3] 页面转换或AI提取的输出图片格式。(默认: png)")
    parser.add_argument("--optimize", action="store_true", help="[模式2/3] 对输出的PNG/WEBP图像进行优化。")

    # --- 模式 2 参数 ---
    parser.add_argument("--parallel", action="store_true", help="[模式2] 启用并行处理加速页面转换。不适用于模式3。")
    
    # --- 模式 1 & 3 参数 ---
    parser.add_argument("-n", "--numbers", type=str, default=None, help="[模式1/3] 仅提取指定序号的图片 (例如 '1,3,5-8')。")
    
    # --- 模式 3 参数 ---
    parser.add_argument("--table", action="store_true", help="[模式3] 启用表格区域提取功能。")
    parser.add_argument("--model-path", type=str, default=None, help="[模式3] ONNX模型文件路径 (.onnx)。")
    parser.add_argument("--imgsz", type=int, default=1024, help="[模式3] 模型推理的输入图像尺寸。(默认: 1024)")
    parser.add_argument("--conf", type=float, default=0.25, help="[模式3] AI模型置信度阈值。(默认: 0.25)")
    parser.add_argument("--device", type=str, default='cpu', help="[模式3] AI推理设备 ('cpu' 或 'cuda')。(默认: 'cpu')")
    parser.add_argument("--padding", type=int, default=0, help="[模式3] 从检测框裁剪时向外的额外填充。(默认: 0)")
    
    args = parser.parse_args()
    
    script_start_time = time.time()
    os.makedirs(args.output, exist_ok=True)
    target_numbers = parse_number_ranges(args.numbers)

    if args.mode == 1:
        extract_embedded_images(args.input, args.output, target_numbers)
    elif args.mode == 2:
        convert_pages_to_images(args.input, args.output, args.dpi, args.parallel, args.optimize, args.format)
    if args.mode == 3:
        # Build a robust path to the default model if not provided
        if not args.model_path:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            model_path = os.path.join(project_root, 'models', 'doclayout_yolo_docstructbench_imgsz1024.onnx')
        else:
            model_path = args.model_path

        if not os.path.exists(model_path):
            parser.error(f"[错误] 模型文件未找到于: {os.path.abspath(model_path)}")
        
        ai_params = {
            "target_numbers": target_numbers, "dpi": args.dpi, "imgsz": args.imgsz,
            "conf": args.conf, "device": args.device, "padding": args.padding,
            "extract_tables": args.table, "out_format": args.format, "optimize": args.optimize
        }
        extract_images_ai_yolo(args.input, args.output, model_path, **ai_params)
    
    total_duration = time.time() - script_start_time
    print("\n========================================")
    print(f"[全局完成] 脚本总耗时: {total_duration:.2f} 秒。")
    print("========================================")

if __name__ == '__main__':
    main()
