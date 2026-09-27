#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
FAQ Markdown 转 HTML 转换脚本

功能：
1. 解析一个或多个 FAQ Markdown 文档，提取结构化数据（支持模块→子模块→错误码分组的分层结构）
2. 支持输入单个 Markdown 文件、多个 Markdown 文件，或一个包含多个 FAQ Markdown 的目录
3. 自动合并多个输入文档，并校验 FAQ ID 唯一性，避免前端展示冲突
4. 仅支持 `modules/faq.md` 所使用的 FAQ 文本格式：
   `标题 / 错误码 / 错误函数 / 关键日志 / 问题现象 / 定位指导 / 图示说明`
5. 将 JSON 数据嵌入到 HTML 模板中，生成最终 HTML 页面

用法：
    1) 单文件：
       python3 generate_faq_html_v2.py faq.md -o index.html

    2) 多文件合并：
       python3 generate_faq_html_v2.py faq.md checker错误码FAQ修正版.md -o full.html

    3) 整目录合并：
       python3 generate_faq_html_v2.py modules/ -o index.html

说明：
    - 目录模式下会自动读取目录中的 *.md 文件，并只纳入真正包含 FAQ 条目的文档。
    - 关键日志、错误码、错误函数、定位指导均按代码块格式解析。
    - HTML 模板默认读取脚本同目录下的 template.html。
"""

import argparse
import re
import sys
import json
from pathlib import Path
from typing import List, Dict


def parse_faq_markdown(md_content: str, source_name: str = '') -> List[Dict]:
    """解析Markdown FAQ文档，提取结构化数据"""
    faq_items = []

    lines = md_content.split('\n')
    current_module = ''
    current_submodule = ''
    current_error_code_group = ''

    faq_blocks = re.split(r'\n#### FAQ-', md_content)

    for i, block in enumerate(faq_blocks[1:]):
        faq_id_match = re.match(r'([A-Z]+\d+)', block)
        if not faq_id_match:
            continue

        faq_id = faq_id_match.group(1)

        original_block = '\n#### FAQ-' + block
        current_module, current_submodule, current_error_code_group = find_context(md_content, original_block)

        faq_item = parse_faq_block(faq_id, block)
        if faq_item:
            faq_item['module'] = current_module
            faq_item['submodule'] = current_submodule
            faq_item['errorCodeGroup'] = current_error_code_group
            faq_item['sourceFile'] = source_name
            faq_items.append(faq_item)

    return faq_items


def find_context(md_content: str, block: str) -> tuple:
    """查找FAQ块在文档中的上下文（模块、子模块、错误码分组）"""
    block_start = md_content.find(block)
    if block_start == -1:
        return '', '', ''

    context_text = md_content[:block_start]
    lines = context_text.split('\n')

    module = ''
    submodule = ''
    error_code_group = ''

    for line in reversed(lines):
        line = line.strip()
        if line.startswith('##### ') and not error_code_group and not submodule:
            match = re.match(r'#####\s*([A-Z_]+)\s*\(\d+\)', line)
            if match:
                error_code_group = match.group(1)
        elif line.startswith('### 子模块：') and not submodule:
            submodule = line.replace('### 子模块：', '').strip()
        elif line.startswith('## 模块：') and not module:
            module = line.replace('## 模块：', '').strip()
            break

    return module, submodule, error_code_group


def extract_error_code_name(error_code_text: str) -> str:
    """从错误码文本中提取错误码名称，支持两种格式：
    1. HCCL_SIM_E_INTERNAL (旧格式)
    2. GRAPH_TRANSLATE_FAILED (101) (V3 格式)
    """
    hccl_match = re.search(r'HCCL_SIM_\w+', error_code_text)
    if hccl_match:
        return hccl_match.group()
    
    v3_match = re.search(r'\b([A-Z][A-Z_]+[A-Z])\s*\(\d+\)', error_code_text)
    if v3_match:
        return v3_match.group(1)
    
    return ''


def parse_faq_block(faq_id: str, block: str) -> Dict:
    """解析单个FAQ块"""
    error_code_blocks = extract_code_block(block, '错误码')
    error_code = '\n'.join(error_code_blocks) if error_code_blocks else ''

    error_functions = extract_error_functions(block)
    guidance = extract_guidance(block)

    faq_item = {
        'id': f'FAQ-{faq_id}',
        'module': '',
        'submodule': '',
        'errorCodeGroup': '',
        'title': extract_field(block, '标题'),
        'errorCode': error_code,
        'errorFunctions': error_functions,
        'keywords': [],
        'logPatterns': extract_code_block(block, '关键日志'),
        'symptom': extract_field(block, '问题现象'),
        'guidance': guidance,
        'diagram': '',
        'diagramType': ''
    }

    if faq_item['errorCode']:
        code_name = extract_error_code_name(faq_item['errorCode'])
        if code_name:
            faq_item['keywords'].append(code_name)

    if faq_item['logPatterns']:
        keywords = extract_keywords_from_log(faq_item['logPatterns'][0])
        faq_item['keywords'].extend(keywords)

    diagram_info = extract_diagram(block)
    faq_item['diagram'] = diagram_info['content']
    faq_item['diagramType'] = diagram_info['type']

    return faq_item


def extract_field(block: str, field_name: str) -> str:
    """提取字段值"""
    pattern = rf'\*\*{field_name}:\*\*\s*(.+)'
    match = re.search(pattern, block)
    return match.group(1).strip() if match else ''


def extract_code_block(block: str, field_name: str) -> List[str]:
    """提取代码块"""
    pattern = rf'\*\*{field_name}:\*\*\s*\n```\s*\n(.*?)```'
    matches = re.findall(pattern, block, re.DOTALL)
    return [m.strip() for m in matches] if matches else []


def extract_keywords_from_log(log_content: str) -> List[str]:
    """从日志中提取关键词"""
    keywords = []
    bracket_keywords = re.findall(r'\[([^\]]+)\]', log_content)
    keywords.extend(bracket_keywords[:3])

    env_vars = re.findall(r'\b[A-Z_]{5,}\b', log_content)
    keywords.extend([v for v in env_vars if 'HCCL' in v or 'RANK' in v])

    return keywords[:5]


def extract_error_functions(block: str) -> List[str]:
    """提取错误函数，仅支持 faq.md 所使用的代码块格式"""
    error_func_blocks = extract_code_block(block, '错误函数')
    error_functions = []
    for func_block in error_func_blocks:
        for line in func_block.split('\n'):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'（.*?）', '', line)
            for item in re.split(r'[、,，]', line):
                item = item.strip()
                if item:
                    error_functions.append(item)
    return error_functions


def extract_guidance(block: str) -> str:
    """提取定位指导部分内容（从 **定位指导:** 后的代码块中）"""
    pattern = r'\*\*定位指导:\*\*\s*\n```\s*\n(.*?)```'
    match = re.search(pattern, block, re.DOTALL)
    if match:
        return match.group(1).strip()
    return ''


def extract_diagram(block: str) -> Dict:
    """提取图示信息"""
    mermaid_pattern = r'\*\*图示说明:\*\*\s*\n```mermaid\s*\n(.*?)```'
    mermaid_match = re.search(mermaid_pattern, block, re.DOTALL)
    if mermaid_match:
        return {'content': mermaid_match.group(1).strip(), 'type': 'mermaid'}

    image_pattern = r'\*\*图示说明:\*\*\s*\n!\[.*?\]\((.*?)\)'
    image_match = re.search(image_pattern, block)
    if image_match:
        return {'content': image_match.group(1), 'type': 'image'}

    return {'content': '', 'type': ''}


def generate_html(faq_data: List[Dict], modules: List[str], submodules: Dict[str, List[str]], error_codes: List[str]) -> str:
    """生成完整的HTML文件内容"""
    html_template = get_html_template()

    json_data = json.dumps(faq_data, ensure_ascii=False, indent=2)
    modules_json = json.dumps(modules, ensure_ascii=False)
    submodules_json = json.dumps(submodules, ensure_ascii=False)
    codes_json = json.dumps(error_codes, ensure_ascii=False)

    html_content = html_template.replace('{{FAQ_DATA}}', json_data)
    html_content = html_content.replace('{{MODULES}}', modules_json)
    html_content = html_content.replace('{{SUBMODULES}}', submodules_json)
    html_content = html_content.replace('{{ERROR_CODES}}', codes_json)

    return html_content


def parse_cli_args(argv: List[str]) -> argparse.Namespace:
    """解析命令行参数，兼容单文件旧用法，也支持传入目录"""
    if len(argv) == 3 and not argv[1].startswith('-'):
        return argparse.Namespace(
            input_md=[argv[1]],
            output_html=argv[2],
        )

    parser = argparse.ArgumentParser(
        description='将一个或多个 FAQ Markdown 文档合并并生成 HTML 查询页面'
    )
    parser.add_argument(
        'input_md',
        nargs='+',
        help='一个或多个 FAQ Markdown 文件或目录路径'
    )
    parser.add_argument(
        '-o', '--output-html',
        required=True,
        help='输出 HTML 文件路径'
    )
    return parser.parse_args(argv[1:])


def expand_input_paths(input_args: List[str]) -> List[Path]:
    """展开输入路径，支持 Markdown 文件和目录"""
    input_files: List[Path] = []
    seen = set()

    for raw_path in input_args:
        path = Path(raw_path)
        if path.is_file():
            resolved = path.resolve()
            if resolved not in seen:
                input_files.append(path)
                seen.add(resolved)
            continue

        if path.is_dir():
            for child in sorted(path.glob('*.md')):
                if not child.is_file():
                    continue
                try:
                    md_content = child.read_text(encoding='utf-8')
                except OSError:
                    continue
                if '\n#### FAQ-' not in md_content:
                    continue
                resolved = child.resolve()
                if resolved in seen:
                    continue
                input_files.append(child)
                seen.add(resolved)
            continue

        input_files.append(path)

    return input_files


def collect_display_metadata(faq_items: List[Dict]) -> tuple[List[str], Dict[str, List[str]], List[str]]:
    """按出现顺序汇总模块、子模块和错误码"""
    modules: List[str] = []
    submodules: Dict[str, List[str]] = {}
    error_codes: List[str] = []

    for item in faq_items:
        module = item.get('module', '')
        submodule = item.get('submodule', '')
        error_code_group = item.get('errorCodeGroup', '')
        error_code = item.get('errorCode', '')

        if module and module not in modules:
            modules.append(module)

        if module and submodule:
            if module not in submodules:
                submodules[module] = []
            if submodule not in submodules[module]:
                submodules[module].append(submodule)

        if error_code:
            code_name = extract_error_code_name(error_code)
            if code_name and code_name not in error_codes:
                error_codes.append(code_name)

        if error_code_group and error_code_group not in error_codes:
            error_codes.append(error_code_group)

    return modules, submodules, error_codes


def validate_unique_faq_ids(faq_items: List[Dict]) -> None:
    """校验 FAQ ID 唯一性，避免多文档合并后出现 DOM 冲突"""
    seen: Dict[str, str] = {}
    duplicates = []

    for item in faq_items:
        faq_id = item.get('id', '')
        source_name = item.get('sourceFile', '') or '<unknown>'
        if faq_id in seen:
            duplicates.append((faq_id, seen[faq_id], source_name))
        else:
            seen[faq_id] = source_name

    if duplicates:
        duplicate_lines = [
            f"  - {faq_id}: {first_source} / {second_source}"
            for faq_id, first_source, second_source in duplicates
        ]
        raise ValueError(
            "发现重复 FAQ ID，多文件合并前请保证每条 FAQ 的编号唯一：\n" +
            "\n".join(duplicate_lines)
        )
def get_html_template() -> str:
    """从外置模板文件读取HTML模板"""
    template_path = Path(__file__).with_name('template.html')
    if not template_path.exists():
        raise FileNotFoundError(f"HTML模板文件不存在: {template_path}")
    return template_path.read_text(encoding='utf-8')


def main():
    args = parse_cli_args(sys.argv)
    input_files = expand_input_paths(args.input_md)
    output_html = args.output_html

    missing_files = [str(path) for path in input_files if not path.exists()]
    if missing_files:
        print("❌ 以下输入文件不存在:")
        for path in missing_files:
            print(f"   - {path}")
        sys.exit(1)

    output_dir = Path(output_html).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    faq_items: List[Dict] = []
    for input_path in input_files:
        print(f"📖 解析: {input_path}")
        md_content = input_path.read_text(encoding='utf-8')
        file_items = parse_faq_markdown(md_content, source_name=input_path.name)
        faq_items.extend(file_items)
        print(f"   提取到 {len(file_items)} 条FAQ")

    try:
        validate_unique_faq_ids(faq_items)
    except ValueError as exc:
        print(f"❌ {exc}")
        sys.exit(1)

    print(f"📚 合计: {len(faq_items)} 条FAQ")

    modules, submodules, error_codes = collect_display_metadata(faq_items)

    print(f"🔧 生成HTML...")
    html_content = generate_html(faq_items, modules, submodules, error_codes)
    Path(output_html).write_text(html_content, encoding='utf-8')

    print(f"✅ 完成: {output_html}")
    print(f"\n请在浏览器中打开查看效果")


if __name__ == '__main__':
    main()
