#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""reader 输入格式适配：BOM 兼容、GBK 回退解码与 zip 成员展开。"""

import zipfile

import pytest

from ttk.core_modules.common.testcase.reader import iter_tables, read_table, resolved_sheet

BOM_CSV = "testcase_name,op_name\r\n case_bom , add\r\n"
PLAIN_CSV = "testcase_name,op_name\r\n case_plain , add\r\n"


def _write_csv(path, content):
    path.write_bytes(content.encode("utf-8"))
    return str(path)


class TestBomHandling:
    def test_bom_csv_header_stripped(self, tmp_path):
        path = _write_csv(tmp_path / "bom.csv", "\ufeff" + BOM_CSV)
        header, rows = read_table(path)
        assert header[0] == "testcase_name"
        assert rows[0][0] == "case_bom"

    def test_plain_csv_unchanged(self, tmp_path, caplog):
        path = _write_csv(tmp_path / "plain.csv", PLAIN_CSV)
        header, rows = read_table(path)
        assert header[0] == "testcase_name"
        assert rows[0][0] == "case_plain"

        # 非 UTF-8（GBK）回退 GB18030：中文表头直读 + warning 留痕
        gbk_path = tmp_path / "gbk.csv"
        gbk_path.write_bytes("testcase_name,注释\n case_gbk , 来源csv文件 \n".encode("gbk"))
        header, rows = read_table(str(gbk_path))
        assert header[1] == "注释"
        assert rows[0][1] == "来源csv文件"
        assert "GB18030" in caplog.text

    def test_zip_member_encodings(self, tmp_path):
        zip_path = tmp_path / "cases.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("bom_member.csv", "\ufeff" + BOM_CSV)
            zf.writestr("gbk_member.csv", "testcase_name,注释\n case_gbk , 来源csv文件 \n".encode("gbk"))
        tables = list(iter_tables(str(zip_path)))
        assert len(tables) == 2
        assert tables[0].header[0] == "testcase_name"
        assert tables[0].rows[0][0] == "case_bom"
        assert tables[1].header[1] == "注释"
        assert tables[1].rows[0][1] == "来源csv文件"

    def test_zip_skips_non_csv_members(self, tmp_path):
        zip_path = tmp_path / "mixed.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("notes.txt", "ignore me")
            zf.writestr("case.csv", PLAIN_CSV)
        tables = list(iter_tables(str(zip_path)))
        assert [t.name for t in tables] == ["case.csv"]

    def test_empty_csv_raises(self, tmp_path):
        path = _write_csv(tmp_path / "empty.csv", "")
        with pytest.raises(ValueError, match="Empty table"):
            read_table(path)


def test_iter_tables_yields_single_table_for_csv(tmp_path):
    path = _write_csv(tmp_path / "case.csv", PLAIN_CSV)
    tables = list(iter_tables(path))
    assert len(tables) == 1
    assert tables[0].name == path
    assert tables[0].header == ["testcase_name", "op_name"]
    assert tables[0].rows == [["case_plain", "add"]]


class TestResolvedSheet:
    """resolved_sheet：输出命名的表名解析（xlsx 首表/显式表；csv/zip → None）。"""

    @pytest.fixture
    def xlsx_path(self, tmp_path):
        openpyxl = pytest.importorskip("openpyxl")
        wb = openpyxl.Workbook()
        wb.active.title = "Alpha"
        wb.create_sheet("Beta")
        path = tmp_path / "cases.xlsx"
        wb.save(path)
        return str(path)

    def test_default_returns_first_sheet(self, xlsx_path):
        assert resolved_sheet(xlsx_path) == "Alpha"

    def test_explicit_sheet(self, xlsx_path):
        assert resolved_sheet(xlsx_path, "Beta") == "Beta"

    def test_non_xlsx_returns_none(self, tmp_path):
        assert resolved_sheet(_write_csv(tmp_path / "case.csv", PLAIN_CSV)) is None
