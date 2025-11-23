# -*- coding: utf-8 -*-
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
from PySide6.QtCore import QObject, Signal


class AppLogic(QObject):
    progress_updated = Signal(int)
    status_changed = Signal(str)
    processing_finished = Signal(str)

    def __init__(self):
        super().__init__()
        self.doc = None

    def run_extraction(self, settings: Dict):
        """
        主执行函数，由线程调用。
        """
        start_time = time.time()
        self.status_changed.emit("任务开始，正在准备环境...")
        self.progress_updated.emit(0) # Start at 0%

        pdf_path = settings["input_path"]
        output_dir = settings["output_dir"]
        mode_text = settings["mode_text"]

        try:
            os.makedirs(output_dir, exist_ok=True)
            self.status_changed.emit("正在打开 PDF 文件...")
            self.progress_updated.emit(2)
            self.doc = fitz.open(pdf_path)
            self.status_changed.emit(f"PDF 文件已打开，共 {len(self.doc)} 页。")
            self.progress_updated.emit(5) # 5% for initial setup

            if "mode 1" in mode_text:
                result = self._extract_mode_1(settings)
            elif "mode 2" in mode_text:
                result = self._convert_mode_2(settings)
            else: # mode 3 is default
                result = self._ai_extract_mode_3(settings)

            self.status_changed.emit("任务完成，正在清理...")
            self.progress_updated.emit(98) # 98% for final cleanup

            duration = time.time() - start_time
            final_message = f"{result} (总耗时: {duration:.2f} 秒)"
            self.processing_finished.emit(final_message)

        except Exception as e:
            self.status_changed.emit(f"发生严重错误: {e}")
            self.processing_finished.emit(f"任务失败: {e}")
        finally:
            if self.doc:
                self.doc.close()
            self.progress_updated.emit(100) # Ensure 100% on completion or error

    def _parse_number_ranges(self, range_string: Optional[str]) -> Optional[Set[int]]:
        if not range_string:
            return None
        selected_numbers = set()
        parts = range_string.split(',')
        for part in parts:
            part = part.strip()
            if not part: continue
            try:
                if '-' in part:
                    start, end = map(int, part.split('-'))
                    selected_numbers.update(range(start, end + 1))
                else:
                    selected_numbers.add(int(part))
            except ValueError:
                self.status_changed.emit(f"[警告] 无法解析序号 '{part}'，将予以忽略。")
        return selected_numbers

    def _extract_mode_1(self, s: Dict) -> str:
        self.status_changed.emit("[模式1] 扫描PDF以提取嵌入的图片对象...")
        target_numbers = self._parse_number_ranges(s["target_numbers"])
        
        all_potential_images = []
        num_pages = len(self.doc)
        for page_index in range(num_pages):
            self.status_changed.emit(f"扫描第 {page_index + 1}/{num_pages} 页...")
            # Progress from 5% to 15% for scanning
            self.progress_updated.emit(int(5 + 10 * (page_index + 1) / num_pages))
            for img_info in self.doc.get_page_images(page_index, full=True):
                rects = self.doc.get_page_image_rects(page_index, img_info)
                if rects:
                    all_potential_images.append({"xref": img_info[0], "bbox": rects[0], "page": page_index})

        if not all_potential_images:
            return "未在文档中找到任何可提取的图片对象。"

        all_potential_images.sort(key=lambda img: (img["page"], img["bbox"].y0, img["bbox"].x0))

        saved_counter = 0
        out_format = s["output_format"]
        total_images_to_save = len(all_potential_images)
        for idx, img_info in enumerate(all_potential_images, 1):
            if target_numbers and idx not in target_numbers:
                continue

            base_image = self.doc.extract_image(img_info["xref"])
            if not base_image or "image" not in base_image:
                continue

            try:
                image_bytes = base_image["image"]
                img = Image.open(io.BytesIO(image_bytes))
                
                if img.mode in ('CMYK', 'P'):
                    img = img.convert('RGB')

                output_path = os.path.join(s["output_dir"], f"fig{idx}.{out_format}")
                
                save_options = {}
                if out_format == 'webp':
                    save_options = {'quality': 95}

                img.save(output_path, format=out_format.upper(), **save_options)
                
                saved_counter += 1
                self.status_changed.emit(f"已保存图片: fig{idx}.{out_format}")
                # Progress from 15% to 90% for saving
                self.progress_updated.emit(int(15 + 75 * (idx + 1) / total_images_to_save))
            except Exception as e:
                self.status_changed.emit(f"警告: 无法转换或保存图片 fig{idx} (xref={img_info['xref']}): {e}")

        return f"成功提取并保存了 {saved_counter} 张图片。"

    def _render_page_worker(self, page_index: int, pdf_path: str, output_dir: str, dpi: int, optimize: bool, out_format: str) -> str:
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
            return f"已保存页面: page_{page_index + 1}.{out_format}"
        except Exception as e:
            return f"错误：处理页面 {page_index + 1} 失败 - {e}"

    def _convert_mode_2(self, s: Dict) -> str:
        self.status_changed.emit(f"[模式2] 开始将PDF页面转换为 {s['output_format'].upper()}...")
        num_pages = len(self.doc)
        
        if s["parallel"]:
            num_cores = min(os.cpu_count() or 1, 8, num_pages)
            self.status_changed.emit(f"运行模式: 并行 (使用 {num_cores} 个核心)")
            with concurrent.futures.ProcessPoolExecutor(max_workers=num_cores) as executor:
                futures = [executor.submit(self._render_page_worker, i, s["input_path"], s["output_dir"], s["dpi"], s["optimize"], s["output_format"]) for i in range(num_pages)]
                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    self.status_changed.emit(future.result())
                    # Progress from 10% to 90% for rendering
                    self.progress_updated.emit(int(10 + 80 * (i + 1) / num_pages))
        else:
            self.status_changed.emit("运行模式: 单线程")
            for i in range(num_pages):
                result = self._render_page_worker(i, s["input_path"], s["output_dir"], s["dpi"], s["optimize"], s["output_format"])
                self.status_changed.emit(result)
                # Progress from 10% to 90% for rendering
                self.progress_updated.emit(int(10 + 80 * (i + 1) / num_pages))

        return f"成功转换并保存了 {num_pages} 个页面。"

    def _ai_extract_mode_3(self, s: Dict) -> str:
        # Build a robust, absolute path to the default model
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        default_model_path = os.path.join(project_root, 'models', 'doclayout_yolo_docstructbench_imgsz1024.onnx')
        
        model_path = s.get("model_path") or default_model_path
        
        if not os.path.exists(model_path):
            return f"错误: 模型文件不存在于 '{os.path.abspath(model_path)}'"

        self.status_changed.emit("正在加载 ONNX 模型...")
        try:
            model = onnx.load(model_path)
            metadata = {prop.key: prop.value for prop in model.metadata_props}
            stride = ast.literal_eval(metadata.get("stride", "32"))
            class_names = ast.literal_eval(metadata.get("names", "{}"))
            self.status_changed.emit(f"模型加载完成: stride={stride}, classes={class_names}")
        except Exception as e:
            return f"错误: 无法加载或解析ONNX模型 '{model_path}': {e}"

        self.progress_updated.emit(0)
        providers = ['CPUExecutionProvider']
        session = onnxruntime.InferenceSession(model_path, providers=providers)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

        self.status_changed.emit(f"[模式3] 运行模式: CPU")
        self.status_changed.emit(f"正在对 {len(self.doc)} 个页面进行AI推理...")

        TABLE_MIN_CONF = s["table_min_conf"]
        TABLE_MIN_ASPECT_RATIO, TABLE_MAX_ASPECT_RATIO, TABLE_MIN_AREA_RATIO = 0.3, 3.0, 0.01

        figures_by_page, tables_by_page = {}, {}

        num_pages = len(self.doc)
        for page_num, page in enumerate(self.doc):
            self.status_changed.emit(f"正在处理第 {page_num + 1}/{num_pages} 页...")

            pix = page.get_pixmap(dpi=100)
            img_cv = cv2.cvtColor(np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n), cv2.COLOR_RGBA2BGR if pix.n == 4 else cv2.COLOR_RGB2BGR)
            tensor, _, pad_info = self._preprocess_image(img_cv, (s['imgsz'], s['imgsz']), stride)
            outputs = session.run([output_name], {input_name: tensor})[0]

            # Post-process detections with Non-Maximal Suppression (NMS)
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

            # Apply NMS per class
            kept_indices = []
            unique_class_ids = np.unique(class_ids)
            for class_id in unique_class_ids:
                class_mask = (class_ids == class_id)
                class_boxes = boxes[class_mask]
                class_scores = scores[class_mask]
                
                # Use a standard IOU threshold for NMS
                iou_threshold = 0.45 
                
                indices_for_class = self._non_max_suppression(class_boxes, class_scores, iou_threshold)
                
                # Map back to original indices
                original_indices = np.where(class_mask)[0]
                kept_indices.extend(original_indices[indices_for_class])

            
            page_figs, page_tbls = [], []
            for i in kept_indices:
                det = raw_detections[i]
                confidence, class_id = det[4], int(det[5])
                cls_name = class_names.get(class_id, '').lower()
                is_figure = cls_name == 'figure'
                is_table = cls_name == 'table' and s['extract_tables']

                if not (is_figure or is_table): continue

                passes_conf = confidence > s['conf']
                if is_table and confidence > TABLE_MIN_CONF: passes_conf = True
                if not passes_conf: continue

                scaled_box = self._scale_boxes((s['imgsz'], s['imgsz']), det[:4][np.newaxis, ...], img_cv.shape[:2], pad_info)[0]

                if is_table:
                    box_w, box_h = scaled_box[2] - scaled_box[0], scaled_box[3] - scaled_box[1]
                    if box_h <= 0 or box_w <= 0: continue
                    aspect_ratio = box_w / box_h
                    area_ratio = (box_w * box_h) / (img_cv.shape[0] * img_cv.shape[1])
                    if not (TABLE_MIN_ASPECT_RATIO < aspect_ratio < TABLE_MAX_ASPECT_RATIO and area_ratio > TABLE_MIN_AREA_RATIO):
                        continue

                page_rect = page.rect
                bbox = fitz.Rect(
                    scaled_box[0] * page_rect.width / img_cv.shape[1], scaled_box[1] * page_rect.height / img_cv.shape[0],
                    scaled_box[2] * page_rect.width / img_cv.shape[1], scaled_box[3] * page_rect.height / img_cv.shape[0]
                )

                result = {'page': page_num, 'bbox': bbox, 'conf': confidence}
                if is_figure: page_figs.append(result)
                elif is_table: page_tbls.append(result)

            if page_figs: figures_by_page[page_num] = page_figs
            if page_tbls: tables_by_page[page_num] = page_tbls

            # Update progress after each page is processed
            progress = int(((page_num + 1) / num_pages) * 95)
            self.progress_updated.emit(progress)

        self.status_changed.emit("推理完成，正在整理和保存结果...")
        all_figures, all_tables = [], []
        for page_num in sorted(figures_by_page.keys()):
            all_figures.extend(self._sort_detected_objects(figures_by_page[page_num], self.doc[page_num].rect))
        if s['extract_tables']:
            for page_num in sorted(tables_by_page.keys()):
                all_tables.extend(self._sort_detected_objects(tables_by_page[page_num], self.doc[page_num].rect))

        target_numbers = self._parse_number_ranges(s["target_numbers"])
        fig_count = self._save_images(all_figures, s["output_dir"], "fig", s["dpi"], s["padding"], s["output_format"], s["optimize_m3"], target_numbers)
        tbl_count = 0
        if s['extract_tables']:
            tbl_count = self._save_images(all_tables, s["output_dir"], "table", s["dpi"], s["padding"], s["output_format"], s["optimize_m3"], None)

        self.progress_updated.emit(100)

        if not all_figures and not all_tables:
            return "AI模型未在此文档中检测到任何'figure'或'table'区域。"

        return f"成功提取了 {fig_count} 张图片和 {tbl_count} 个表格。"


    def _save_images(self, objects: List[Dict], output_dir: str, prefix: str, dpi: int, padding: int, out_format: str, optimize: bool, target_numbers: Optional[Set[int]]):
        if not objects: return 0
            
        self.status_changed.emit(f"AI模型共检测到 {len(objects)} 个有效{prefix}区域，正在保存...")
        saved_count = 0
        for idx, obj in enumerate(objects, 1):
            if target_numbers and idx not in target_numbers: continue
            
            bbox = obj['bbox']
            if bbox.is_empty or bbox.width <= 10 or bbox.height <= 10: continue
                
            clip_rect = (bbox + (-padding, -padding, padding, padding)).irect
            pix = self.doc[obj['page']].get_pixmap(clip=clip_rect, dpi=dpi)
            
            output_path = os.path.join(output_dir, f"{prefix}{idx}.{out_format}")
            
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            save_options = {}
            if out_format == 'webp':
                save_options['quality'] = 95
                if optimize:
                    save_options['method'] = 6
            img.save(output_path, format=out_format.upper(), **save_options)

            saved_count += 1
            self.status_changed.emit(f"已保存图片: {prefix}{idx}.{out_format}")

        self.status_changed.emit(f"成功保存了 {saved_count} 张{prefix}。")
        return saved_count

    def _preprocess_image(self, image: np.ndarray, new_shape: Tuple[int, int], stride: int) -> Tuple[np.ndarray, float, Tuple[int, int]]:
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

    def _scale_boxes(self, img1_shape: Tuple[int, int], boxes: np.ndarray, img0_shape: Tuple[int, int], pad_info: Tuple[int, int]) -> np.ndarray:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad_x, pad_y = pad_info
        boxes[..., 0] = (boxes[..., 0] - pad_x) / gain
        boxes[..., 1] = (boxes[..., 1] - pad_y) / gain
        boxes[..., 2] = (boxes[..., 2] - pad_x) / gain
        boxes[..., 3] = (boxes[..., 3] - pad_y) / gain
        boxes[..., [0, 2]] = np.clip(boxes[..., [0, 2]], 0, img0_shape[1])
        boxes[..., [1, 3]] = np.clip(boxes[..., [1, 3]], 0, img0_shape[0])
        return boxes

    def _calculate_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Calculates the Intersection over Union (IoU) of two bounding boxes."""
        # box format: [x1, y1, x2, y2]
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

    def _non_max_suppression(self, boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> List[int]:
        """Performs Non-Maximal Suppression to remove overlapping bounding boxes."""
        if len(boxes) == 0:
            return []

        # Sort by score in descending order
        sorted_indices = np.argsort(scores)[::-1]

        kept_indices = []
        while len(sorted_indices) > 0:
            # Pick the box with the highest score
            current_index = sorted_indices[0]
            kept_indices.append(current_index)

            # Get the coordinates of the current box
            current_box = boxes[current_index]

            # Get the rest of the boxes
            remaining_indices = sorted_indices[1:]
            
            if len(remaining_indices) == 0:
                break

            # Calculate IoU of the current box with the rest
            ious = np.array([self._calculate_iou(current_box, boxes[i]) for i in remaining_indices])

            # Keep boxes with IoU less than the threshold
            # Note: The indices in 'below_threshold_indices' are relative to the 'remaining_indices' array
            below_threshold_indices = np.where(ious < iou_threshold)[0]

            # Update sorted_indices to be the ones that were kept
            sorted_indices = remaining_indices[below_threshold_indices]
            
        return kept_indices


    def _sorted_boxes(self, objects: List[Dict]) -> List[Dict]:
        """
        Sort boxes in order from top to bottom, left to right, inspired by MinerU.
        """
        if not objects:
            return []

        # Sort primarily by top coordinate (y0), then by left coordinate (x0)
        sorted_objects = sorted(objects, key=lambda obj: (obj['bbox'].y0, obj['bbox'].x0))
        
        # Refine sorting for items on the same line (small y0 difference)
        i = 0
        while i < len(sorted_objects) - 1:
            # Find a group of items on the same "line"
            line_end_j = i
            for j in range(i + 1, len(sorted_objects)):
                # Check if the vertical distance is small enough to be considered the same line
                if abs(sorted_objects[j]['bbox'].y0 - sorted_objects[i]['bbox'].y0) < 20:
                    line_end_j = j
                else:
                    break
            
            # If a line group is found, sort it by x-coordinate
            if line_end_j > i:
                line_group = sorted_objects[i : line_end_j + 1]
                sorted_line_group = sorted(line_group, key=lambda obj: obj['bbox'].x0)
                sorted_objects[i : line_end_j + 1] = sorted_line_group
                i = line_end_j + 1
            else:
                i += 1
                
        return sorted_objects

    def _sort_detected_objects(self, objects: List[Dict], page_rect: fitz.Rect) -> List[Dict]:
        # The new sorting logic is more robust and doesn't need to distinguish between spanning and columnar objects.
        return self._sorted_boxes(objects)
