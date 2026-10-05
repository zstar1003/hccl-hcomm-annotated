#!/usr/bin/env python3
"""Apply reviewed standalone annotations and build exact-line AllReduce guides.

The manifest records code from an immutable, already annotated snapshot. Checks
bind every note to that code, require full code-line coverage of selected ranges,
and verify current source against a pinned publication revision. No dependencies.
"""
import argparse
import collections
import html
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'docs/allreduce/line-notes.json'
MARKER = '// [中文导读] '
LINE_MARKER = re.compile(r'^\s*// \[中文导读\] \[AllReduce逐行 S(\d+)\] (.+)\n?$')
SNAPSHOTS = {}


def git(*args):
    return subprocess.check_output(['git', *args], cwd=ROOT)


def snapshot(base, path):
    key = (base, path)
    if key not in SNAPSHOTS:
        SNAPSHOTS[key] = git('show', f'{base}:{path}').decode().splitlines(keepends=True)
    return SNAPSHOTS[key]


def classify(lines):
    """Conservative C++ lexical scan: code lines and safe insertion positions."""
    mode, quote, raw_end = 'code', '', ''
    code, safe = [], []
    previous = ''
    for line in lines:
        safe.append(mode == 'code' and not re.search(r'\\[\t ]*\r?\n$', previous))
        meaningful = False
        i = 0
        while i < len(line):
            char = line[i]
            following = line[i + 1:i + 2]
            if mode == 'block':
                if char == '*' and following == '/':
                    mode = 'code'
                    i += 2
                else:
                    i += 1
                continue
            if mode == 'raw':
                meaningful = True
                if line.startswith(raw_end, i):
                    i += len(raw_end)
                    mode = 'code'
                else:
                    i += 1
                continue
            if mode == 'literal':
                meaningful = True
                if char == '\\':
                    i += 2
                elif char == quote:
                    mode = 'code'
                    i += 1
                else:
                    i += 1
                continue
            if char == '/' and following == '/':
                break
            if char == '/' and following == '*':
                mode = 'block'
                i += 2
                continue
            raw = re.match(r'(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(', line[i:])
            if raw:
                meaningful = True
                mode, raw_end = 'raw', ')' + raw.group(1) + '"'
                i += len(raw.group())
                continue
            # Apostrophes in digit separators do not open character literals.
            number = re.match(r'[0-9][\w.\x27]*', line[i:]) if char.isdigit() else None
            if number:
                meaningful = True
                i += len(number.group())
                continue
            if char in ('"', "'"):
                meaningful = True
                mode, quote = 'literal', char
            elif not char.isspace():
                meaningful = True
            i += 1
        code.append(meaningful)
        previous = line
    return code, safe


def validate_manifest(data):
    notes = {}
    for note in data['notes']:
        path, line = note['path'], note['line']
        if not re.fullmatch(r'(hccl|hcomm)/src/[\w/.-]+\.(cc|h|hpp)', path):
            raise ValueError(f'Unexpected source path {path}')
        original = snapshot(data['base'], path)
        if original[line - 1].rstrip('\r\n') != note['code']:
            raise ValueError(f'Snapshot/code mismatch: {path}:{line}')
        if not note['note'].strip() or '\n' in note['note']:
            raise ValueError(f'Invalid note: {path}:{line}')
        key = (path, line)
        if key in notes:
            raise ValueError(f'Duplicate note: {path}:{line}')
        notes[key] = note
    required = set()
    owners = {}
    for fn in data['functions']:
        path = fn['path']
        lines = snapshot(data['base'], path)
        flags, _ = classify(lines)
        if not 1 <= fn['start'] <= fn['end'] <= len(lines):
            raise ValueError(f'Invalid selected range: {path}:{fn["start"]}')
        if 'REGISTER_' not in fn['name']:
            leaf = re.sub(r'\s+(?:const|noexcept)\b.*$', '', fn['name'].rsplit('::', 1)[-1])
            head = ''.join(lines[fn['start'] - 1:min(fn['start'] + 5, fn['end'])])
            if not re.search(re.escape(leaf) + r'\s*\(', head):
                raise ValueError(f'Function signature missing at selected start: {path}:{fn["start"]} {fn["name"]}')
        for number in range(fn['start'], fn['end'] + 1):
            if (path, number) in owners:
                raise ValueError(f'Overlapping function ranges: {path}:{number} {owners[(path, number)]} / {fn["name"]}')
            owners[(path, number)] = fn['name']
            if flags[number - 1]:
                required.add((path, number))
    missing = required - notes.keys()
    if missing:
        raise ValueError(f'{len(missing)} uncovered code lines: {sorted(missing)[:15]}')
    extras = notes.keys() - required
    if extras:
        raise ValueError(f'Notes outside selected code ranges: {sorted(extras)[:15]}')
    return notes


def source_maps(data, notes, apply=False):
    mappings = {}
    by_path = collections.defaultdict(list)
    for note in notes.values():
        by_path[note['path']].append(note)
    for path, entries in sorted(by_path.items()):
        original = snapshot(data['base'], path)
        current = (ROOT / path).read_bytes().decode().splitlines(keepends=True)
        retained = [line for line in current if not LINE_MARKER.match(line)]
        if retained != original:
            raise ValueError(f'Unreviewed source edits since snapshot: {path}')
        _, safe = classify(original)
        inserts = collections.defaultdict(list)
        for note in sorted(entries, key=lambda item: item['line']):
            position = note['line'] - 1
            while not safe[position]:
                position -= 1
                if position < 0:
                    raise ValueError(f'No safe insertion for {path}:{note["line"]}')
            indent = re.match(r'[\t ]*', original[position]).group()
            inserts[position].append(f'{indent}{MARKER}[AllReduce逐行 S{note["line"]}] {note["note"]}\n')
        expected, mapping = [], {}
        for idx, line in enumerate(original):
            expected.extend(inserts[idx])
            mapping[idx + 1] = len(expected) + 1
            expected.append(line)
        if apply:
            (ROOT / path).write_bytes(''.join(expected).encode())
        elif current != expected:
            raise ValueError(f'Missing or stale source annotations: {path}; use --apply')
        mappings[path] = mapping
        revision = data.get('sourceRevision', 'WORKTREE')
        if revision != 'WORKTREE' and git('show', f'{revision}:{path}') != ''.join(expected).encode():
            raise ValueError(f'Published revision differs from current source: {path}')
    return mappings


def line_map(data, mappings, path):
    if path in mappings:
        return mappings[path]
    original = snapshot(data['base'], path)
    current = (ROOT / path).read_bytes().decode().splitlines(keepends=True)
    mapping = {}
    idx = 0
    for number, text in enumerate(current, 1):
        if text.startswith(MARKER) or text.lstrip().startswith(MARKER):
            if idx < len(original) and text == original[idx]:
                idx += 1
                mapping[idx] = number
            continue
        if idx >= len(original) or text != original[idx]:
            raise ValueError(f'Unmapped reference file: {path}:{number}')
        idx += 1
        mapping[idx] = number
    if idx != len(original):
        raise ValueError(f'Incomplete reference map: {path}')
    mappings[path] = mapping
    return mapping


def link(data, maps, path, start, end=None, label=None):
    mapping = line_map(data, maps, path)
    first, last = mapping[start], mapping[end or start]
    revision = data.get('sourceRevision', 'WORKTREE')
    revision = 'main' if revision == 'WORKTREE' else revision
    url = f'https://github.com/zstar1003/hccl-hcomm-annotated/blob/{revision}/{path}#L{first}'
    if last != first:
        url += f'-L{last}'
    title = label or f'{Path(path).name}:L{first}' + (f'–L{last}' if last != first else '')
    return f'[{title}]({url})'


def render(data, maps):
    groups = [
        ('01-entry-selection', '入口、引擎与算法选择'),
        ('02-dispatch-device', '公共调度、资源上下文与设备入口'),
        ('03-hcomm-resources', 'HCOMM通信资源与建链'),
        ('04-data-plane', 'OneShot编排、传输包装与底层任务提交'),
    ]
    notes = {(n['path'], n['line']): n for n in data['notes']}
    outputs = {}
    total_rows = 0
    for slug, title in groups:
        funcs = [fn for fn in data['functions'] if fn['group'] == slug]
        introduction = [f'# AllReduce逐行对照：{title}\n',
                 '[返回阅读指南](../READING_GUIDE.zh-CN.md)。S为审读快照行号；L为带本次逐行注释的源码行号。'
                 '每个L链接定位到固定源码提交；长语句按物理行分别说明。空行及原注释不重复注释。\n',
                 f'审读快照：`{data["base"]}`；源码提交：`{data.get("sourceRevision", "WORKTREE")}`。\n']
        blocks = []
        for idx, fn in enumerate(funcs, 1):
            parts = [f'## {idx}. {fn["name"]}\n',
                      f'{fn["purpose"]}\n',
                      f'完整范围：{link(data, maps, fn["path"], fn["start"], fn["end"])}；'
                      f'文件：`{fn["path"]}`。\n']
            if fn.get('branches'):
                parts += ['功能与分支：\n']
                for branch in fn['branches']:
                    parts += [f'- {branch}\n']
                parts += ['\n']
            table = ['| 源码定位 | 原代码 | 这一行的功能 |', '|---|---|---|']
            for number in range(fn['start'], fn['end'] + 1):
                note = notes.get((fn['path'], number))
                if not note:
                    continue
                code = '<code>' + html.escape(note['code']).replace('|', '&#124;') + '</code>'
                explain = html.escape(note['note']).replace('|', '&#124;')
                location = link(data, maps, fn['path'], number, label=f'S{number} / L{maps[fn["path"]][number]}')
                table += [f'| {location} | {code} | {explain} |']
                total_rows += 1
            parts += ['\n'.join(table), '\n']
            blocks.append((fn, '\n'.join(parts)))
        # Keep Markdown pages comfortably small and preserve complete functions.
        chunks = []
        chunk = []
        size = 0
        for fn, block in blocks:
            block_size = len(block.encode())
            if chunk and size + block_size > 250_000:
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append((fn, block))
            size += block_size
        if chunk:
            chunks.append(chunk)
        index = introduction + ['按完整函数拆页，表中页链接打开逐行代码/说明，源码链接直接定位带注释的固定提交。\n']
        rows = []
        for page, chunk in enumerate(chunks, 1):
            filename = f'{slug}.part-{page:02}.zh-CN.md'
            outputs[f'docs/allreduce/{filename}'] = '\n'.join(introduction + [
                f'[返回本阶段函数导航]({slug}.zh-CN.md)。第{page}/{len(chunks)}页。\n'] +
                [block for _, block in chunk])
            for fn, _ in chunk:
                rows += [f'| `{fn["name"]}` | [第{page}页]({filename}) | '
                         f'{link(data, maps, fn["path"], fn["start"], fn["end"])} |']
        index += ['| 函数/范围 | 逐行页 | 固定源码 |\n|---|---|---|\n' + '\n'.join(rows)]
        outputs[f'docs/allreduce/{slug}.zh-CN.md'] = '\n'.join(index)
    template = (ROOT / 'docs/allreduce/READING_GUIDE.template.md').read_text()
    def resolve_token(match):
        path, first, last, label = match.group(1).split('|')
        path = data.get('aliases', {}).get(path, path)
        return link(data, maps, path, int(first), int(last) if last else None, label or None)
    template = re.sub(r'\{\{src\|([^}]+)\}\}', resolve_token, template)
    template = template.replace('{{noteCount}}', str(len(notes))).replace('{{functionCount}}', str(len(data['functions'])))
    template = template.replace('{{sourceRevision}}', data.get('sourceRevision', 'WORKTREE'))
    outputs['docs/READING_GUIDE.zh-CN.md'] = template
    relations = ['# AllReduce分阶段调用关系与分支\n',
                 '[返回阅读指南](../READING_GUIDE.zh-CN.md)。以下关系由固定快照逐函数审读；'
                 '路径定位已转换为带逐行注释的固定提交行号。Host→AICPU为runtime发射关系，'
                 '虚调用按本例注册的executor/template/UB传输类型展开。'
                 '树内简写行号对应审读快照S行；树后定位表和正文链接对应新增注释后的L行。\n']
    relation_index = list(relations)
    function_ends = {(fn['path'], fn['start']): fn['end'] for fn in data['functions']}
    for graph_number, graph in enumerate(data.get('graphs', []), 1):
        body = graph['body']
        references = {}
        def resolve_evidence(match, in_fence=False):
            path, first, last = match.group(1), int(match.group(2)), match.group(3)
            last = int(last) if last else None
            if last and (path, first) in function_ends:
                last = function_ends[(path, first)]
            if in_fence:
                references[(path, first, last)] = None
                return f'{Path(path).name}:S{first}' + (f'–S{last}' if last else '')
            return link(data, maps, path, first, last)
        pattern = r'`?((?:hccl|hcomm)/[\w/.-]+):(\d+)(?:[–-](\d+))?`?'
        segments = re.split(r'(```[^\n]*\n[\s\S]*?```)', body)
        body = ''.join(re.sub(pattern, lambda m: resolve_evidence(m, part.startswith('```')), part)
                       for part in segments)
        if references:
            body += '\n\n### 调用树中的精确源码定位\n\n| 审读快照位置 | 带逐行注释的固定源码 |\n|---|---|\n'
            for (path, first, last) in references:
                coordinate = f'{path}:S{first}' + (f'–S{last}' if last else '')
                body += f'| `{coordinate}` | {link(data, maps, path, first, last)} |\n'
        sections = re.split(r'(?m)(?=^#{1,4} )', body)
        chunks, chunk, size = [], [], 0
        for section in sections:
            if chunk and size + len(section.encode()) > 220_000:
                chunks.append(chunk)
                chunk, size = [], 0
            chunk.append(section)
            size += len(section.encode())
        if chunk:
            chunks.append(chunk)
        relation_index += [f'## {graph["title"]}\n']
        for page, chunk in enumerate(chunks, 1):
            filename = f'CALL_RELATIONS-{graph_number}.part-{page:02}.zh-CN.md'
            outputs[f'docs/allreduce/{filename}'] = '\n'.join(relations + [
                f'[返回调用关系导航](CALL_RELATIONS.zh-CN.md)。{graph["title"]}，第{page}/{len(chunks)}页。\n',
                ''.join(chunk)])
            headings = [re.sub(r'^#+ ', '', section.splitlines()[0]) for section in chunk if section.startswith('#')]
            relation_index += [f'- [第{page}页]({filename})：' + '；'.join(headings[:5]) +
                               (f'；另有{len(headings) - 5}节' if len(headings) > 5 else '') + '\n']
        relation_index += ['\n']
    outputs['docs/allreduce/CALL_RELATIONS.zh-CN.md'] = '\n'.join(relation_index)
    outputs['docs/allreduce/COVERAGE.zh-CN.md'] = (
        '# AllReduce逐行覆盖范围\n\n'
        f'审读快照：`{data["base"]}`；固定源码提交：`{data.get("sourceRevision", "WORKTREE")}`。\n\n'
        f'共{len(notes)}条逐物理行说明、{len(data["functions"])}个选定函数/代码范围、'
        f'{len({fn["path"] for fn in data["functions"]})}个文件。'
        '覆盖单位是列明范围内的非空、非原注释代码行，含签名、结构行、预处理和长语句续行。'
        '签名/注册范围也可能单列，因此范围数不等于独立函数数。'
        '这不是全仓或所有AllReduce算法的注释覆盖声明。\n\n'
        '| 文件 | 函数/范围 | 带注释源码定位 | 逐行数 |\n|---|---|---|---|\n' +
        ''.join(f'| `{fn["path"]}` | `{fn["name"]}` | '
                f'{link(data, maps, fn["path"], fn["start"], fn["end"])} | '
                f'{sum((fn["path"], n) in notes for n in range(fn["start"], fn["end"] + 1))} |\n'
                for fn in data['functions']) +
        '\n## 有意停止展开的边界\n\n' +
        '\n'.join(f'- {boundary}' for boundary in data['boundaries']) + '\n\n'
        '机器清单见[line-notes.json](line-notes.json)。生成器逐条核对快照行内容、所选范围代码行覆盖、'
        '插入注释和固定提交字节；主仓校验器另证去掉全部中文导读后恢复上游基线。\n'
    )
    return outputs, total_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='insert exact reviewed standalone notes')
    parser.add_argument('--check', action='store_true', help='verify sources, coverage and generated guides without writing')
    args = parser.parse_args()
    if args.apply and args.check:
        parser.error('--apply and --check are mutually exclusive')
    data = json.loads(MANIFEST.read_text())
    notes = validate_manifest(data)
    maps = source_maps(data, notes, apply=args.apply)
    outputs, rows = render(data, maps)
    for path, content in outputs.items():
        if args.check:
            if not (ROOT / path).exists() or (ROOT / path).read_text() != content:
                raise ValueError(f'Stale generated guide: {path}')
        else:
            (ROOT / path).write_text(content)
    for path in (ROOT / 'docs/allreduce').glob('*.part-*.zh-CN.md'):
        if str(path.relative_to(ROOT)) not in outputs:
            if args.check:
                raise ValueError(f'Obsolete generated page: {path}')
            path.unlink()
    print(json.dumps({'notes': len(notes), 'ranges': len(data['functions']), 'tableRows': rows,
                      'files': len({fn['path'] for fn in data['functions']}), 'guides': len(outputs)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
