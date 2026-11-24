import os
import tkinter as tk
from tkinter import filedialog
from . import functions


class ImageProcessorGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("영상 처리 프로그램")
        self.root.geometry("1050x700")

        self.origin_img = None
        self.result_img = None
        self.tk_origin = None
        self.tk_result = None

        self.container = tk.Frame(self.root)
        self.container.pack(fill=tk.BOTH, expand=True)

        self.menu_tree = {
            "회색 이미지 변환": [
                ("기본 그레이스케일", functions.gray_basic),
                ("흑백 반전", functions.gray_invert),
                ("이진화 (임계값 127)", functions.gray_binary),
            ],
            "컬러 이미지 변환": [
                ("빨강", functions.color_red),
                ("초록", functions.color_green),
                ("파랑", functions.color_blue),
                ("컬러 (원본)", functions.color_original),
            ],
            "블러링": [
                ("평균 블러링", functions.blur_average),
                ("가우시안 블러링", functions.blur_gaussian),
                ("박스 필터 블러링", functions.blur_box),
            ],
            "샤프닝": [
                ("언샤프 마스크", functions.sharpen_unsharp),
                ("라플라시안 샤프닝", functions.sharpen_laplacian),
                ("강한 커스텀 샤프닝", functions.sharpen_strong),
            ],
            "에지 검출": [
                ("소벨(Sobel)", functions.edge_sobel),
                ("라플라시안(Laplacian)", functions.edge_laplacian),
                ("캐니(Canny)", functions.edge_canny),
            ],
            "필터링": [
                ("최솟값 필터", functions.filter_min),
                ("최댓값 필터", functions.filter_max),
                ("미디언 필터", functions.filter_median),
                ("주파수 저주파통과(LPF)", functions.filter_freq_lpf),
                ("주파수 고주파통과(HPF)", functions.filter_freq_hpf),
            ],
            "모폴로지": [
                ("침식 (Erode)", functions.morph_erode),
                ("팽창 (Dilate)", functions.morph_dilate),
                ("열림 (Open)", functions.morph_open),
                ("닫힘 (Close)", functions.morph_close),
            ],
            "확대/축소": [
                ("0.5배 축소", functions.scale_down_half),
                ("1.5배 확대", functions.scale_up_1_5),
                ("2.0배 확대", functions.scale_up_2_0),
            ],
            "보간법": [
                ("최근접 이웃 (Nearest)", functions.interp_nearest),
                ("양선형 (Linear)", functions.interp_linear),
                ("3차 회선 (Cubic)", functions.interp_cubic),
            ],
            "평행이동": [
                ("오른쪽 이동 (+60px)", functions.translate_right),
                ("아래쪽 이동 (+60px)", functions.translate_down),
                ("대각선 이동 (+50, +50)", functions.translate_diag),
            ],
            "회전": [
                ("90도 회전", functions.rotate_90),
                ("180도 회전", functions.rotate_180),
                ("270도 회전", functions.rotate_270),
                ("45도 회전", functions.rotate_45),
            ],
            "명암 대비 조절": [
                ("밝기 증가 (+50)", functions.bright_up),
                ("밝기 감소 (-50)", functions.bright_down),
                ("대비 증가 (1.6배)", functions.contrast_up),
                ("대비 감소 (0.5배)", functions.contrast_down),
                ("히스토그램 평활화", functions.contrast_hist_eq),
            ],
            "합성": [
                ("이미지 선택 (5 : 5 합성)", lambda img: self._blend_action(img, 0.5)),
                ("이미지 선택 (7 : 3 합성)", lambda img: self._blend_action(img, 0.7)),
                ("이미지 선택 (3 : 7 합성)", lambda img: self._blend_action(img, 0.3)),
            ],
        }

        self._show_start_screen()

    def _clear_widget(self, widget):
        for child in widget.winfo_children():
            child.destroy()

    def _show_start_screen(self):
        self._clear_widget(self.container)

        center_box = tk.Frame(self.container)
        center_box.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

        tk.Button(
            center_box,
            text="파일 가져오기",
            command=self.open_file_dialog,
            width=16,
            height=2,
            font=("맑은 고딕", 11),
        ).pack()

        self.error_label = tk.Label(
            center_box,
            text="",
            fg="red",
            font=("맑은 고딕", 10),
        )
        self.error_label.pack(pady=(12, 0))

    def open_file_dialog(self):
        path = filedialog.askopenfilename(
            title="이미지 파일 선택",
            filetypes=[
                ("이미지 파일 (png, jpg)", "*.png *.jpg"),
                ("모든 파일", "*.*"),
            ],
        )

        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        if ext not in (".png", ".jpg"):
            self.error_label.config(
                text="오류: png, jpg 파일만 선택할 수 있습니다. 파일을 다시 선택해주세요."
            )
            return

        img = functions.load_image(path)
        if img is None:
            self.error_label.config(
                text="오류: 이미지를 불러올 수 없습니다. 파일을 다시 선택해주세요."
            )
            return

        self.origin_img = img
        self.result_img = img.copy()
        self._show_workspace_screen()

    def _show_workspace_screen(self):
        self._clear_widget(self.container)

        top_frame = tk.Frame(self.container, bg="#252526", pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        left_panel = tk.Frame(top_frame, bg="#252526")
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(
            left_panel, text="[ 원본 이미지 ]", fg="#cccccc", bg="#252526", font=("맑은 고딕", 10, "bold")
        ).pack(pady=(0, 6))
        self.origin_label = tk.Label(left_panel, bg="#1e1e1e")
        self.origin_label.pack(expand=True, fill=tk.BOTH)

        right_panel = tk.Frame(top_frame, bg="#252526")
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        tk.Label(
            right_panel, text="[ 변환 이미지 (결과) ]", fg="#4fc3f7", bg="#252526", font=("맑은 고딕", 10, "bold")
        ).pack(pady=(0, 6))
        self.result_label = tk.Label(right_panel, bg="#1e1e1e")
        self.result_label.pack(expand=True, fill=tk.BOTH)

        self.tk_origin = functions.cv_to_tk(self.origin_img)
        self.origin_label.config(image=self.tk_origin)
        self._update_result_display(self.result_img)

        self.control_frame = tk.Frame(self.container, bg="#f0f0f0", height=175)
        self.control_frame.pack(side=tk.BOTTOM, fill=tk.X)
        self.control_frame.pack_propagate(False)

        self._show_category_buttons()

    def _show_category_buttons(self):
        self._clear_widget(self.control_frame)

        btn_box = tk.Frame(self.control_frame, bg="#f0f0f0")
        btn_box.pack(expand=True)

        categories = list(self.menu_tree.keys())
        cols = 5
        for idx, cat_name in enumerate(categories):
            r, c = divmod(idx, cols)
            tk.Button(
                btn_box,
                text=cat_name,
                width=18,
                height=1,
                font=("맑은 고딕", 10),
                command=lambda name=cat_name: self._show_sub_buttons(name),
            ).grid(row=r, column=c, padx=6, pady=5)

        r, c = divmod(len(categories), cols)
        tk.Button(
            btn_box,
            text="결과 초기화",
            width=18,
            height=1,
            bg="#dcdcdc",
            font=("맑은 고딕", 10),
            command=self._reset_result,
        ).grid(row=r, column=c, padx=6, pady=5)

    def _show_sub_buttons(self, category_name: str):
        self._clear_widget(self.control_frame)

        header_bar = tk.Frame(self.control_frame, bg="#e4e4e4", pady=6, padx=15)
        header_bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(
            header_bar,
            text="뒤로가기",
            width=10,
            bg="#d0d0d0",
            font=("맑은 고딕", 9, "bold"),
            command=self._show_category_buttons,
        ).pack(side=tk.LEFT)

        tk.Label(
            header_bar,
            text=f"선택된 기능 : {category_name}",
            bg="#e4e4e4",
            font=("맑은 고딕", 10, "bold"),
        ).pack(side=tk.LEFT, padx=15)

        sub_box = tk.Frame(self.control_frame, bg="#f0f0f0")
        sub_box.pack(expand=True)

        sub_items = self.menu_tree[category_name]
        for opt_name, func in sub_items:
            tk.Button(
                sub_box,
                text=opt_name,
                width=20,
                height=2,
                font=("맑은 고딕", 10),
                command=lambda f=func: self._apply_sub_function(f),
            ).pack(side=tk.LEFT, padx=8)

    def _apply_sub_function(self, func):
        if self.origin_img is None:
            return
        res = func(self.origin_img)
        if res is not None:
            self.result_img = res
            self._update_result_display(self.result_img)

    def _blend_action(self, img, alpha: float):
        path = filedialog.askopenfilename(
            title="합성할 두 번째 이미지 선택",
            filetypes=[("이미지 파일 (png, jpg)", "*.png *.jpg")],
        )
        if not path:
            return None
        other = functions.load_image(path)
        if other is None:
            return None
        return functions.blend_images(img, other, alpha)

    def _update_result_display(self, img):
        self.tk_result = functions.cv_to_tk(img)
        self.result_label.config(image=self.tk_result)

    def _reset_result(self):
        if self.origin_img is None:
            return
        self.result_img = self.origin_img.copy()
        self._update_result_display(self.result_img)