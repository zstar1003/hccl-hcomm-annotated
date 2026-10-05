// Repository-local checks; no network access, dependencies, or source rewriting.
import { execFileSync } from 'node:child_process';
import { readFileSync, lstatSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const marker = '// [中文导读] ';
const annotation = /^[\t ]*\/\/ \[中文导读\] [^\r\n]+\r?\n$/;

// A conservative lexical guard, not a C++ compiler. Reject insertion into strings,
// comments, or continued lines; exact byte restoration remains the primary check.
function scanLine(line, state) {
  for (let i = 0; i < line.length; i++) {
    const c = line[i], next = line[i + 1];
    if (state.mode === 'line') break;
    if (state.mode === 'block') {
      if (c === '*' && next === '/') { state.mode = 'code'; i++; }
      continue;
    }
    if (state.mode === 'raw') {
      if (line.startsWith(state.end, i)) { i += state.end.length - 1; state.mode = 'code'; }
      continue;
    }
    if (state.mode === 'literal') {
      if (c === '\\') { i++; continue; }
      if (c === state.quote) state.mode = 'code';
      continue;
    }
    if (c === '/' && next === '/') { state.mode = 'line'; break; }
    if (c === '/' && next === '*') { state.mode = 'block'; i++; continue; }
    const raw = line.slice(i).match(/^(?:u8|u|U|L)?R"([^\s()\\]{0,16})\(/);
    if (raw) { state.mode = 'raw'; state.end = `)${raw[1]}"`; i += raw[0].length - 1; continue; }
    if (/\d/.test(c) && !/[\w]/.test(line[i - 1] ?? '')) {
      const number = line.slice(i).match(/^[\d][\w.']*/);
      if (number) { i += number[0].length - 1; continue; }
    }
    if (c === '"' || c === "'") { state.mode = 'literal'; state.quote = c; }
  }
  if (state.mode === 'line' && !/\\\r?\n$/.test(line)) state.mode = 'code';
}

export function verifyCommentOnly(original, current, name = '<input>') {
  if (original.includes(marker)) throw new Error(`${name}: baseline already contains reserved marker`);
  const text = current.toString('utf8');
  if (!Buffer.from(text, 'utf8').equals(current)) throw new Error(`${name}: invalid UTF-8`);
  const lines = text.match(/[^\n]*\n|[^\n]+$/g) ?? [];
  const retained = [];
  const state = { mode: 'code' };
  let previous = '', count = 0, blocks = 0, wasAnnotation = false;
  for (const line of lines) {
    if (line.includes(marker)) {
      if (!annotation.test(line)) throw new Error(`${name}: marker must be a complete standalone comment line`);
      if (state.mode !== 'code') throw new Error(`${name}: annotation inside ${state.mode}`);
      if (/\\[\t ]*\r?\n$/.test(previous) || /\\[\t ]*\r?\n$/.test(line)) {
        throw new Error(`${name}: annotation touches a continued line`);
      }
      count++;
      if (!wasAnnotation) blocks++;
      wasAnnotation = true;
    } else {
      retained.push(line);
      scanLine(line, state);
      previous = line;
      wasAnnotation = false;
    }
  }
  if (!Buffer.from(retained.join(''), 'utf8').equals(original)) {
    throw new Error(`${name}: original bytes changed, deleted, or unmarked content added`);
  }
  if (!count) throw new Error(`${name}: declared file has no added annotation`);
  return { lines: count, blocks };
}

export function verifyRepository(root) {
  const git = (...args) => execFileSync('git', args, { cwd: root, maxBuffer: 32 * 1024 * 1024 });
  const text = (...args) => git(...args).toString('utf8').trim();
  const coverage = JSON.parse(readFileSync(resolve(root, 'annotations.json'), 'utf8'));
  const sources = JSON.parse(readFileSync(resolve(root, 'sources.lock.json'), 'utf8'));
  const baseline = coverage.baseline;
  if (!/^[a-f0-9]{40}$/.test(baseline)) throw new Error('Expected immutable baseline commit SHA');
  const declared = coverage.files.map(item => item.path).sort();
  if (new Set(declared).size !== declared.length) throw new Error('Duplicate coverage path');

  const totals = {};
  for (const project of ['hccl', 'hcomm']) {
    if (text('rev-parse', `${baseline}:${project}`) !== sources[project].tree) {
      throw new Error(`${project}: baseline does not match locked upstream tree`);
    }
    const files = git('ls-tree', '-rz', '--name-only', baseline, '--', project).toString('utf8').split('\0').filter(Boolean);
    totals[project] = files.length;
  }

  const changed = git('diff', '--name-only', '-z', baseline, '--', 'hccl', 'hcomm').toString('utf8').split('\0').filter(Boolean).sort();
  if (JSON.stringify(changed) !== JSON.stringify(declared)) {
    throw new Error('Changed upstream paths must exactly equal annotations.json files');
  }
  const untracked = text('ls-files', '--others', '--exclude-standard', '--', 'hccl', 'hcomm');
  if (untracked) throw new Error('Untracked additions found inside upstream directories');
  const results = [];
  for (const path of declared) {
    if (!/^(hccl|hcomm)\/src\/[\w/.-]+\.(cc|h|hpp)$/.test(path)) throw new Error(`Unexpected annotation path: ${path}`);
    const stat = lstatSync(resolve(root, path));
    if (!stat.isFile() || stat.isSymbolicLink()) throw new Error(`${path}: not a regular source file`);
    const originalMode = text('ls-tree', baseline, '--', path).split(' ')[0];
    const mode = stat.mode & 0o111 ? '100755' : '100644';
    if (mode !== originalMode) throw new Error(`${path}: file mode changed`);
    results.push({ path, ...verifyCommentOnly(git('show', `${baseline}:${path}`), readFileSync(resolve(root, path)), path) });
  }
  // Also catches unwanted whitespace changes in root-authored documentation.
  git('diff', '--check', baseline);
  return {
    baseline, upstreamFiles: totals, annotatedFiles: results.length,
    annotationLines: results.reduce((n, r) => n + r.lines, 0),
    annotationBlocks: results.reduce((n, r) => n + r.blocks, 0), files: results,
    limitation: 'Text-preservation and insertion checks only; not compilation, runtime, or binary-equivalence proof.'
  };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const result = verifyRepository(resolve(dirname(fileURLToPath(import.meta.url)), '..'));
    process.stdout.write(`${JSON.stringify(result, null, 2)}\n`);
  } catch (error) {
    process.stderr.write(`FAIL: ${error.message}\n`);
    process.exitCode = 1;
  }
}
