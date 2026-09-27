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
1. 解析 Markdown FAQ 文档，提取结构化数据（支持模块→子模块→错误码分组的分层结构）
2. 生成 JSON 数据
3. 将 JSON 嵌入到 HTML 模板中，生成最终的 index.html

用法：
    python3 scripts/generate_faq_html.py docs/faq/test_faq.md docs/faq/index.html
"""

import re
import sys
import json
from pathlib import Path
from typing import List, Dict


def parse_faq_markdown(md_content: str) -> List[Dict]:
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


def parse_faq_block(faq_id: str, block: str) -> Dict:
    """解析单个FAQ块"""
    # 错误码和错误函数现在使用代码块格式
    error_code_blocks = extract_code_block(block, '错误码')
    error_code = '\n'.join(error_code_blocks) if error_code_blocks else extract_field(block, '错误码')

    error_func_blocks = extract_code_block(block, '错误函数')
    if error_func_blocks:
        # 将多行合并为逗号分隔的列表
        error_functions = []
        for b in error_func_blocks:
            for line in b.split('\n'):
                line = line.strip()
                if line:
                    # 去掉括号内的中文描述
                    line = re.sub(r'（.*?）', '', line)
                    # 按逗号、顿号、中文逗号拆分
                    for item in re.split(r'[、,，]', line):
                        item = item.strip()
                        if item:
                            error_functions.append(item)
    else:
        error_functions = extract_list_field(block, '错误函数')

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
        'guidance': extract_guidance(block),
        'diagram': '',
        'diagramType': ''
    }

    if faq_item['errorCode']:
        code_match = re.search(r'HCCL_SIM_\w+', faq_item['errorCode'])
        if code_match:
            faq_item['keywords'].append(code_match.group())

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


def extract_list_field(block: str, field_name: str) -> List[str]:
    """提取逗号分隔的列表字段"""
    value = extract_field(block, field_name)
    if not value:
        return []
    return [v.strip() for v in value.split(',') if v.strip()]


def extract_code_block(block: str, field_name: str) -> List[str]:
    """提取代码块"""
    pattern = rf'\*\*{field_name}:\*\*\s*\n```.*?\n(.*?)```'
    matches = re.findall(pattern, block, re.DOTALL)
    return [m.strip() for m in matches] if matches else []


def extract_numbered_list(block: str, field_name: str) -> List[str]:
    """提取编号列表"""
    pattern = rf'\*\*{field_name}:\*\*\s*\n((?:\d+\..+\n)+)'
    match = re.search(pattern, block)
    if not match:
        return []

    list_text = match.group(1)
    items = re.findall(r'\d+\.\s*(.+)', list_text)
    return [item.strip() for item in items]


def extract_keywords_from_log(log_content: str) -> List[str]:
    """从日志中提取关键词"""
    keywords = []
    bracket_keywords = re.findall(r'\[([^\]]+)\]', log_content)
    keywords.extend(bracket_keywords[:3])

    env_vars = re.findall(r'\b[A-Z_]{5,}\b', log_content)
    keywords.extend([v for v in env_vars if 'HCCL' in v or 'RANK' in v])

    return keywords[:5]


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


def get_html_template() -> str:
    """获取HTML模板"""
    return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HCCL-VM FAQ 查询系统</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh; padding: 20px;
        }
        .container {
            max-width: 1200px; margin: 0 auto;
            background: #fff; border-radius: 12px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3); overflow: hidden;
        }
        .header {
            background: linear-gradient(90deg, #2c3e50, #3498db);
            color: white; padding: 30px; text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 10px; }
        .header p { opacity: 0.9; font-size: 14px; }
        .search-panel { background: #f8f9fa; padding: 25px; border-bottom: 1px solid #dee2e6; }
        .search-row { display: flex; gap: 15px; margin-bottom: 15px; flex-wrap: wrap; }
        .search-group { flex: 1; min-width: 180px; }
        .search-group label { display: block; font-weight: 600; margin-bottom: 8px; color: #2c3e50; }
        .search-group select, .search-group input {
            width: 100%; padding: 10px 15px; border: 2px solid #e0e0e0;
            border-radius: 8px; font-size: 14px; transition: all 0.3s;
        }
        .search-group select:focus, .search-group input:focus {
            border-color: #3498db; outline: none;
            box-shadow: 0 0 0 3px rgba(52, 152, 219, 0.1);
        }
        .btn-row { display: flex; gap: 10px; justify-content: center; }
        .btn { padding: 12px 30px; border: none; border-radius: 8px;
               font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.3s; }
        .btn-primary { background: #3498db; color: white; }
        .btn-primary:hover { background: #2980b9; transform: translateY(-2px); }
        .btn-secondary { background: #95a5a6; color: white; }
        .btn-secondary:hover { background: #7f8c8d; }
        .stats-bar { background: #ecf0f1; padding: 15px 25px;
                     display: flex; justify-content: space-between; align-items: center;
                     font-size: 13px; color: #7f8c8d; }
        .results-container { padding: 25px; min-height: 400px; }
        .no-results { text-align: center; padding: 60px; color: #95a5a6; }
        .no-results h3 { font-size: 20px; margin-bottom: 10px; }
        .faq-item { background: #fff; border: 1px solid #e0e0e0; border-radius: 10px;
                    margin-bottom: 20px; overflow: hidden; transition: all 0.3s; }
        .faq-item:hover { box-shadow: 0 5px 15px rgba(0,0,0,0.1); border-color: #3498db; }
        .faq-header { background: linear-gradient(90deg, #f8f9fa, #e9ecef); padding: 15px 20px;
                      cursor: pointer; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px; }
        .faq-header:hover { background: linear-gradient(90deg, #ecf0f1, #d5dbdb); }
        .faq-id { font-weight: 700; color: #3498db; font-size: 14px; }
        .faq-title { flex: 1; margin-left: 15px; font-weight: 600; color: #2c3e50; min-width: 200px; }
        .faq-tags { display: flex; gap: 8px; flex-wrap: wrap; }
        .tag { padding: 4px 10px; border-radius: 12px; font-size: 12px; font-weight: 500; }
        .tag-module { background: #e8f5e9; color: #2e7d32; }
        .tag-submodule { background: #e3f2fd; color: #1565c0; }
        .tag-code { background: #fff3e0; color: #e65100; }
        .faq-content { padding: 20px; display: none; }
        .faq-content.active { display: block; }
        .faq-section { margin-bottom: 15px; }
        .faq-section h4 { font-size: 14px; color: #34495e; margin-bottom: 8px; font-weight: 600; }
        .faq-section p, .faq-section ul { font-size: 13px; color: #555; line-height: 1.6; }
        .faq-section ul { padding-left: 20px; }
        .faq-section li { margin-bottom: 5px; }
        .code-block { background: #2d3436; color: #dfe6e9; padding: 15px; border-radius: 6px;
                      font-family: "Consolas", "Monaco", monospace; font-size: 13px;
                      overflow-x: auto; white-space: pre-wrap; }
        .diagram-container { background: #f8f9fa; padding: 15px; border-radius: 6px;
                             margin-top: 10px; text-align: center; }
        .diagram-container img { max-width: 100%; border-radius: 6px; }
        .expand-icon { font-size: 18px; color: #95a5a6; transition: transform 0.3s; }
        .expand-icon.rotated { transform: rotate(180deg); }
        .highlight { background: #fff176; padding: 2px 4px; border-radius: 3px; }
        .module-group-header {
            background: linear-gradient(90deg, #2c3e50, #3498db); color: white;
            padding: 14px 20px; font-size: 17px; font-weight: 700;
            margin: 25px 0 12px 0; border-radius: 8px;
            display: flex; align-items: center; gap: 10px;
        }
        .module-group-header:first-child { margin-top: 0; }
        .module-count {
            background: rgba(255,255,255,0.25); padding: 2px 10px;
            border-radius: 10px; font-size: 12px; font-weight: 500;
        }
        .submodule-group-header {
            background: #f0f4f8; border-left: 4px solid #3498db;
            padding: 10px 16px; margin: 16px 0 8px 8px; border-radius: 0 6px 6px 0;
            cursor: pointer; display: flex; align-items: center;
            gap: 8px; transition: background 0.2s; user-select: none;
        }
        .submodule-group-header:hover { background: #e2e8f0; }
        .submodule-group-header .sub-title { font-size: 15px; font-weight: 600; color: #2c3e50; }
        .submodule-group-header .sub-count {
            background: #3498db; color: white; padding: 2px 8px;
            border-radius: 10px; font-size: 11px; font-weight: 500;
        }
        .submodule-group-header .sub-toggle { color: #7f8c8d; font-size: 12px; margin-left: auto; }
        .submodule-group.collapsed .faq-item { display: none; }
        .submodule-group.collapsed .submodule-group-header { opacity: 0.65; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>HCCL-VM 常见问题查询系统</h1>
            <p>按模块、子模块、错误码、关键词快速定位问题 | 支持模糊匹配</p>
        </div>
        <div class="search-panel">
            <div class="search-row">
                <div class="search-group">
                    <label>按模块筛选</label>
                    <select id="moduleFilter" onchange="onModuleChange()"><option value="">全部模块</option></select>
                </div>
                <div class="search-group">
                    <label>按子模块筛选</label>
                    <select id="submoduleFilter"><option value="">全部子模块</option></select>
                </div>
                <div class="search-group">
                    <label>按错误码筛选</label>
                    <select id="errorCodeFilter"><option value="">全部错误码</option></select>
                </div>
            </div>
            <div class="search-row">
                <div class="search-group" style="flex: 2;">
                    <label>关键词搜索</label>
                    <input type="text" id="keywordInput"
                           placeholder="输入关键词：函数名、日志内容、问题描述...">
                </div>
            </div>
            <div class="btn-row">
                <button class="btn btn-primary" onclick="searchFAQ()">🔍 搜索</button>
                <button class="btn btn-secondary" onclick="clearFilters()">🔄 清空</button>
            </div>
        </div>
        <div class="stats-bar">
            <span id="statsInfo">共 <strong>0</strong> 条FAQ</span>
            <span id="matchInfo"></span>
        </div>
        <div class="results-container" id="resultsContainer">
            <div class="no-results">
                <h3>请输入搜索条件</h3>
                <p>选择模块、子模块或错误码，或输入关键词进行搜索</p>
            </div>
        </div>
    </div>

    <script>
        const faqData = {{FAQ_DATA}};
        const modules = {{MODULES}};
        const submodules = {{SUBMODULES}};
        const errorCodes = {{ERROR_CODES}};
        let currentKeyword = '';

        mermaid.initialize({ startOnLoad: false, theme: 'neutral' });

        function initModuleFilter() {
            const select = document.getElementById('moduleFilter');
            modules.forEach(m => {
                const option = document.createElement('option');
                option.value = m; option.textContent = m;
                select.appendChild(option);
            });
        }

        function initSubmoduleFilter(moduleName) {
            const select = document.getElementById('submoduleFilter');
            select.innerHTML = '<option value="">全部子模块</option>';
            if (!moduleName || !submodules[moduleName]) return;
            submodules[moduleName].forEach(sm => {
                const option = document.createElement('option');
                option.value = sm; option.textContent = sm;
                select.appendChild(option);
            });
        }

        function initErrorCodeFilter() {
            const select = document.getElementById('errorCodeFilter');
            errorCodes.forEach(c => {
                const option = document.createElement('option');
                option.value = c; option.textContent = c;
                select.appendChild(option);
            });
        }

        function onModuleChange() {
            const moduleName = document.getElementById('moduleFilter').value;
            initSubmoduleFilter(moduleName);
            document.getElementById('submoduleFilter').value = '';
            searchFAQ();
        }

        function searchFAQ() {
            const moduleFilter = document.getElementById('moduleFilter').value;
            const submoduleFilter = document.getElementById('submoduleFilter').value;
            const errorCodeFilter = document.getElementById('errorCodeFilter').value;
            const keyword = document.getElementById('keywordInput').value.trim().toLowerCase();
            currentKeyword = keyword;

            let results = faqData.filter(item => {
                if (moduleFilter && item.module !== moduleFilter) return false;
                if (submoduleFilter && item.submodule !== submoduleFilter) return false;
                if (errorCodeFilter) {
                    if (item.errorCode && item.errorCode.includes(errorCodeFilter)) return true;
                    if (item.errorCodeGroup && item.errorCodeGroup === errorCodeFilter) return true;
                    return false;
                }
                if (keyword) {
                    const searchFields = [
                        item.title, item.errorCode, item.errorCodeGroup, ...item.errorFunctions,
                        ...item.keywords, ...item.logPatterns, item.symptom,
                        item.guidance, item.id, item.module, item.submodule
                    ].join(' ').toLowerCase();
                    return searchFields.includes(keyword);
                }
                return true;
            });
            displayResults(results);
        }

        function displayResults(results) {
            const container = document.getElementById('resultsContainer');
            document.getElementById('statsInfo').innerHTML = `共 <strong>${faqData.length}</strong> 条FAQ`;
            document.getElementById('matchInfo').innerHTML = results.length > 0 ?
                `匹配到 <strong>${results.length}</strong> 条结果` : '';

            if (results.length === 0) {
                container.innerHTML = '<div class="no-results"><h3>未找到匹配结果</h3><p>请调整筛选条件</p></div>';
                return;
            }

            // 按模块 → 子模块分组
            const grouped = {};
            results.forEach(item => {
                const mod = item.module || '未知模块';
                const sub = item.submodule || '其他';
                if (!grouped[mod]) grouped[mod] = {};
                if (!grouped[mod][sub]) grouped[mod][sub] = [];
                grouped[mod][sub].push(item);
            });

            let html = '';
            const moduleOrder = modules.filter(m => grouped[m]);
            Object.keys(grouped).filter(m => !modules.includes(m)).forEach(m => moduleOrder.push(m));

            moduleOrder.forEach(mod => {
                const moduleCount = Object.values(grouped[mod]).reduce((sum, arr) => sum + arr.length, 0);
                html += `<div class="module-group-header">
                    <span>${mod}</span>
                    <span class="module-count">${moduleCount} 条</span>
                </div>`;

                const submoduleOrder = (submodules[mod] || []).filter(sm => grouped[mod][sm]);
                Object.keys(grouped[mod]).filter(sm => !(submodules[mod] || []).includes(sm)).forEach(sm => submoduleOrder.push(sm));

                submoduleOrder.forEach(sub => {
                    const items = grouped[mod][sub];
                    const groupId = `group-${mod}-${sub}`.replace(/[\\s（）()]/g, '_');
                    html += `<div class="submodule-group" id="${groupId}">
                        <div class="submodule-group-header" onclick="toggleSubmoduleGroup('${groupId}')">
                            <span class="sub-title">${sub}</span>
                            <span class="sub-count">${items.length} 条</span>
                            <span class="sub-toggle">收起/展开</span>
                        </div>`;

                    items.forEach(item => {
                        html += `<div class="faq-item">
                            <div class="faq-header" onclick="toggleFAQ('${item.id}')">
                                <span class="faq-id">${item.id}</span>
                                <span class="faq-title">${highlightText(item.title)}</span>
                                <div class="faq-tags">
                                    ${item.errorCode ? `<span class="tag tag-code">${extractErrorCode(item.errorCode)}</span>` : (item.errorCodeGroup ? `<span class="tag tag-code">${item.errorCodeGroup}</span>` : '')}
                                </div>
                                <span class="expand-icon" id="icon-${item.id}">▼</span>
                            </div>
                            <div class="faq-content" id="content-${item.id}">${renderFAQContent(item)}</div>
                        </div>`;
                    });
                    html += '</div>';
                });
            });
            container.innerHTML = html;

            setTimeout(() => {
                document.querySelectorAll('.mermaid').forEach(el => mermaid.run({ nodes: [el] }));
            }, 100);
        }

        function toggleSubmoduleGroup(groupId) {
            const group = document.getElementById(groupId);
            if (group) group.classList.toggle('collapsed');
        }

        function renderFAQContent(item) {
            let html = '';
            if (item.errorCode) html += `<div class="faq-section"><h4>错误码</h4><p>${item.errorCode}</p></div>`;
            if (item.errorFunctions && item.errorFunctions.length > 0)
                html += `<div class="faq-section"><h4>错误函数</h4><ul>${item.errorFunctions.map(f => `<li>${f}</li>`).join('')}</ul></div>`;
            if (item.logPatterns && item.logPatterns.length > 0)
                html += `<div class="faq-section"><h4>关键日志</h4><div class="code-block">${highlightText(item.logPatterns[0])}</div></div>`;
            if (item.symptom) html += `<div class="faq-section"><h4>问题现象</h4><p>${highlightText(item.symptom)}</p></div>`;
            if (item.guidance) html += `<div class="faq-section"><h4>定位指导</h4><div class="code-block">${highlightText(item.guidance)}</div></div>`;
            if (item.diagram)
                html += `<div class="faq-section"><h4>图示说明</h4><div class="diagram-container">${item.diagramType === 'mermaid' ? `<pre class="mermaid">${item.diagram}</pre>` : `<img src="${item.diagram}" alt="示意图">`}</div></div>`;
            return html;
        }

        function toggleFAQ(id) {
            const content = document.getElementById('content-' + id);
            const icon = document.getElementById('icon-' + id);
            content.classList.toggle('active');
            icon.classList.toggle('rotated');
        }

        function highlightText(text) {
            if (!currentKeyword || !text) return text;
            const regex = new RegExp(currentKeyword, 'gi');
            return text.replace(regex, match => `<span class="highlight">${match}</span>`);
        }

        function extractErrorCode(codeStr) {
            const match = codeStr.match(/HCCL_SIM_\w+/);
            return match ? match[0] : codeStr;
        }

        function clearFilters() {
            document.getElementById('moduleFilter').value = '';
            document.getElementById('submoduleFilter').innerHTML = '<option value="">全部子模块</option>';
            document.getElementById('errorCodeFilter').value = '';
            document.getElementById('keywordInput').value = '';
            currentKeyword = '';
            displayResults(faqData);
        }

        window.onload = function() {
            initModuleFilter(); initErrorCodeFilter(); displayResults(faqData);
        };

        document.getElementById('keywordInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') searchFAQ();
        });
    </script>
</body>
</html>'''


def main():
    if len(sys.argv) < 3:
        print("用法: python3 scripts/generate_faq_html.py <input_md> <output_html>")
        print("示例: python3 scripts/generate_faq_html.py docs/faq/test_faq.md docs/faq/index.html")
        sys.exit(1)

    input_md = sys.argv[1]
    output_html = sys.argv[2]

    if not Path(input_md).exists():
        print(f"❌ 输入文件不存在: {input_md}")
        sys.exit(1)

    output_dir = Path(output_html).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"📖 解析: {input_md}")
    md_content = Path(input_md).read_text(encoding='utf-8')

    faq_items = parse_faq_markdown(md_content)

    print(f"   提取到 {len(faq_items)} 条FAQ")

    modules = sorted(set(item['module'] for item in faq_items if item['module']))

    submodules = {}
    for item in faq_items:
        if item['module'] and item['submodule']:
            if item['module'] not in submodules:
                submodules[item['module']] = []
            if item['submodule'] not in submodules[item['module']]:
                submodules[item['module']].append(item['submodule'])
    for key in submodules:
        submodules[key] = sorted(submodules[key])

    error_codes = []
    for item in faq_items:
        if item['errorCode']:
            match = re.search(r'HCCL_SIM_\w+', item['errorCode'])
            if match and match.group() not in error_codes:
                error_codes.append(match.group())
        if item['errorCodeGroup'] and item['errorCodeGroup'] not in error_codes:
            error_codes.append(item['errorCodeGroup'])
    error_codes = sorted(error_codes)

    print(f"🔧 生成HTML...")
    html_content = generate_html(faq_items, modules, submodules, error_codes)
    Path(output_html).write_text(html_content, encoding='utf-8')

    print(f"✅ 完成: {output_html}")
    print(f"\n请在浏览器中打开查看效果")


if __name__ == '__main__':
    main()