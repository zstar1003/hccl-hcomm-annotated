import test from 'node:test';
import assert from 'node:assert/strict';
import { verifyCommentOnly } from './verify-annotations.mjs';

const b = text => Buffer.from(text, 'utf8');
const comment = '// [中文导读] 说明参数与处理。\n';
const original = 'int f() {\n    return 1;\n}\n';

test('accepts only new tagged full-line comments and counts contiguous blocks', () => {
  assert.deepEqual(verifyCommentOnly(b(original), b(comment + comment + original)), { lines: 2, blocks: 1 });
});
test('rejects executable changes', () => {
  assert.throws(() => verifyCommentOnly(b(original), b(comment + original.replace('return 1', 'return 2'))));
});
test('rejects deleted code and changed original comments', () => {
  assert.throws(() => verifyCommentOnly(b(original), b(comment + 'int f() {}\n')));
  assert.throws(() => verifyCommentOnly(b('// Copyright original\n' + original), b(comment + '// Copyright changed\n' + original)));
});
test('rejects inline marker, untagged insertion, and missing newline', () => {
  assert.throws(() => verifyCommentOnly(b(original), b('int x; ' + comment + original)));
  assert.throws(() => verifyCommentOnly(b(original), b('// plain\n' + comment + original)));
  assert.throws(() => verifyCommentOnly(b(original), b(original + comment.trimEnd())));
});
test('rejects macro continuation and comment line-splicing', () => {
  const macro = '#define X \\\n  1\n';
  assert.throws(() => verifyCommentOnly(b(macro), b(macro.replace('  1', comment + '  1'))));
  assert.throws(() => verifyCommentOnly(b(original), b(comment.replace('。\n', '。\\\n') + original)));
});
test('rejects insertion inside multiline literal or block comment', () => {
  for (const source of ['const char* x = R"tag(\nbody\n)tag";\n', '/*\nbody\n*/\n', 'const char* x = "a\\\nbody";\n']) {
    assert.throws(() => verifyCommentOnly(b(source), b(source.replace('body', comment + 'body'))));
  }
});
test('allows insertion after complete raw string, number separator, and ordinary comment', () => {
  const source = 'auto n = 1\'000; // a value\nconst char* s = R"(// not a comment)";\n' + original;
  assert.deepEqual(verifyCommentOnly(b(source), b(source.replace(original, comment + original))), { lines: 1, blocks: 1 });
});
test('preserves CRLF bytes, rejects baseline marker and empty change', () => {
  const crlf = original.replaceAll('\n', '\r\n');
  assert.equal(verifyCommentOnly(b(crlf), b(comment.replaceAll('\n', '\r\n') + crlf)).lines, 1);
  assert.throws(() => verifyCommentOnly(b(crlf), b(comment + original)));
  assert.throws(() => verifyCommentOnly(b(comment + original), b(comment + original)));
  assert.throws(() => verifyCommentOnly(b(original), b(original)));
});
