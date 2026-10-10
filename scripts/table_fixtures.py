"""表格结构验证用的合成表格 PDF。

为什么要自己造样本：本机带表格的真实 PDF 只有 hak180 一份，单样本不足以说明
「行列结构还原可用」，而且真实表格的真值要靠人读图。用 PIL 画表格再存 PDF，
真值就是我写进单元格的文本，可以逐格比对而不是「看着像对」。
"""

from __future__ import annotations

from dataclasses import dataclass, field

FONT = "C:/Windows/Fonts/msyh.ttc"
PAGE_W, PAGE_H = 1240, 1754  # A4 @150dpi，与真实扫描页同一尺度


@dataclass
class TableSpec:
    """一张表：可选的合并表头行 + 列名行 + 数据行。"""

    name: str
    title: str
    header: list[str]
    rows: list[list[str]]
    top: tuple[str, str] | None = None  # (首列文本, 跨列文本)，跨列文本吃掉 len(header)-1 列
    col0_weight: float = 2.0

    @property
    def n_cols(self) -> int:
        return len(self.header)

    @property
    def grid(self) -> list[list[tuple[str, int]]]:
        """真值网格：每格是 (文本, 跨列数)。"""
        out: list[list[tuple[str, int]]] = []
        if self.top:
            out.append([(self.top[0], 1), (self.top[1], self.n_cols - 1)])
        out.append([(c, 1) for c in self.header])
        out.extend([(c, 1) for c in r] for r in self.rows)
        return out

    @property
    def n_rows(self) -> int:
        return len(self.grid)


@dataclass
class Fixture:
    path: str
    spec: TableSpec
    texts: list[str] = field(default_factory=list)


def _weights(spec: TableSpec) -> list[float]:
    return [spec.col0_weight] + [1.0] * (spec.n_cols - 1)


def render_pdf(spec: TableSpec, out_path: str, top_y: int = 180, row_h: int = 48) -> str:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    d = ImageDraw.Draw(img)
    f_title = ImageFont.truetype(FONT, 38)
    f_cell = ImageFont.truetype(FONT, 24)
    d.text((110, 84), spec.title, font=f_title, fill="black")

    x0, x1 = 110, 1130
    ws = _weights(spec)
    total = sum(ws)
    edges = [x0]
    for w in ws:
        edges.append(edges[-1] + int((x1 - x0) * w / total))
    edges[-1] = x1

    y = top_y
    for r_i, row in enumerate(spec.grid):
        h = row_h if r_i else max(row_h, 56)
        ci = 0
        for text, span in row:
            left, right = edges[ci], edges[min(len(edges) - 1, ci + span)]
            ci += span
            d.rectangle([left, y, right, y + h], outline="black", width=3)
            if text:
                bb = d.textbbox((0, 0), text, font=f_cell)
                d.text(
                    ((left + right) / 2 - (bb[2] - bb[0]) / 2, y + h / 2 - (bb[3] - bb[1]) / 2 - bb[1]),
                    text,
                    font=f_cell,
                    fill="black",
                )
        y += h
    d.text((110, y + 40), "注：以上数据取自出厂检验记录。", font=f_cell, fill="black")
    img.save(out_path, "PDF", resolution=150.0)
    return out_path


SPECS: list[TableSpec] = [
    TableSpec(
        name="普通三列表",
        title="表 1 设备环境适应性要求",
        header=["项目", "要求值", "判定"],
        rows=[
            ["高温工作", "40 摄氏度", "合格"],
            ["低温存储", "零下 20 摄氏度", "合格"],
            ["湿度范围", "10% 至 90%", "合格"],
            ["振动等级", "5 级", "待复测"],
            ["防护等级", "IP54", "合格"],
        ],
    ),
    TableSpec(
        name="合并表头七列表",
        title="表 2 物料有害物质含量标识",
        top=("型号", "有害物质"),
        header=["部件名称", "铅", "汞", "镉", "六价铬", "封装剂", "阻燃剂"],
        rows=[
            ["主板单元", "超", "未超", "未超", "未超", "未超", "未超"],
            ["电源模块", "超", "未超", "未超", "未超", "未超", "未超"],
            ["外壳组件", "未超", "未超", "未超", "未超", "未超", "未超"],
            ["线缆组件", "超", "未超", "未超", "未超", "未超", "未超"],
        ],
    ),
    TableSpec(
        name="两列长表",
        title="表 3 里程碑与交付物对照",
        header=["阶段", "交付物"],
        rows=[
            ["需求确认", "需求规格说明书"],
            ["方案设计", "技术方案与接口约定"],
            ["开发联调", "联调记录与缺陷清单"],
            ["试运行", "试运行报告"],
            ["验收", "验收结论与遗留问题"],
            ["结项", "结项报告与复盘记录"],
            ["质保", "质保期服务承诺"],
        ],
    ),
    TableSpec(
        name="宽表六列",
        title="表 4 各区域能耗与成本对照",
        header=["区域", "年产量", "单位能耗", "电费单价", "年成本", "同比"],
        rows=[
            ["华东", "1200 台", "8.4 千瓦时", "0.62 元", "74 万元", "下降 6%"],
            ["华南", "980 台", "9.1 千瓦时", "0.70 元", "69 万元", "持平"],
            ["西南", "640 台", "10.3 千瓦时", "0.48 元", "34 万元", "下降 12%"],
        ],
    ),
]


def build_all(out_dir: str) -> list[Fixture]:
    """生成全部合成表格 PDF，返回 (路径, 规格)。"""
    out_dir = out_dir.replace("\\", "/")
    return [Fixture(path=render_pdf(s, f"{out_dir}/表格_{s.name}.pdf"), spec=s) for s in SPECS]


if __name__ == "__main__":
    import sys

    for fx in build_all(sys.argv[1] if len(sys.argv) > 1 else "var/m1/fixtures"):
        print(fx.path)
