"""Regression checks for exact-line coverage and safe annotation placement."""
import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('guide', Path(__file__).with_name('build-allreduce-guide.py'))
guide = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guide)


class GuideChecks(unittest.TestCase):
    def test_code_after_block_comment_and_inline_comment_requires_note(self):
        lines = ['/* original\n', 'comment */ int x = 1; // suffix\n', '\n', '// old note\n']
        flags, safe = guide.classify(lines)
        self.assertEqual(flags, [False, True, False, False])
        self.assertFalse(safe[1])

    def test_continued_macro_and_raw_literal_are_not_safe_to_split(self):
        lines = ['#define F(x) \\\n', '    use(x)\n', 'auto s = R"tag(first\n', 'second)tag";\n']
        flags, safe = guide.classify(lines)
        self.assertEqual(flags, [True, True, True, True])
        self.assertEqual(safe, [True, False, True, False])

    def test_apostrophe_digit_separator_does_not_open_a_string(self):
        flags, safe = guide.classify(["int n = 1'024;\n", 'consume(n);\n'])
        self.assertEqual(flags, [True, True])
        self.assertEqual(safe, [True, True])

    def manifest(self, notes):
        return {'base': 'a' * 40, 'functions': [
            {'path': 'hccl/src/a.cc', 'name': 'REGISTER_FIXTURE', 'start': 1, 'end': 2}], 'notes': notes}

    def test_missing_physical_code_line_and_wrong_snapshot_are_rejected(self):
        source = ['int x = 1;\n', 'return x;\n']
        notes = [{'path': 'hccl/src/a.cc', 'line': 1, 'code': 'int x = 1;', 'note': '保存结果。'}]
        with patch.object(guide, 'snapshot', return_value=source):
            with self.assertRaisesRegex(ValueError, 'uncovered code lines'):
                guide.validate_manifest(self.manifest(notes))
            notes[0]['code'] = 'int x = 2;'
            with self.assertRaisesRegex(ValueError, 'Snapshot/code mismatch'):
                guide.validate_manifest(self.manifest(notes))

    def test_continued_line_notes_move_before_macro_and_are_idempotent(self):
        source = ['#define F(x) \\\n', '    use(x)\n']
        notes = [{'path': 'hccl/src/a.cc', 'line': n, 'code': source[n - 1].rstrip('\n'),
                  'note': '定义调用。' if n == 1 else '宏主体使用参数。'} for n in (1, 2)]
        data = self.manifest(notes)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / 'hccl/src/a.cc'
            path.parent.mkdir(parents=True)
            path.write_text(''.join(source))
            with patch.object(guide, 'ROOT', root), patch.object(guide, 'snapshot', return_value=source):
                required = guide.validate_manifest(data)
                maps = guide.source_maps(data, required, apply=True)
                self.assertEqual(maps['hccl/src/a.cc'], {1: 3, 2: 4})
                output = path.read_text()
                self.assertIn('#define F(x) \\\n    use(x)\n', output)
                self.assertEqual(len(output.splitlines()), 4)
                guide.source_maps(data, required, apply=True)
                self.assertEqual(path.read_text(), output)
                guide.source_maps(data, required)
                path.write_text(output.replace('宏主体使用参数。', '错改注释。'))
                with self.assertRaisesRegex(ValueError, 'stale source annotations'):
                    guide.source_maps(data, required)

    def test_overlapping_or_incorrect_function_ranges_are_rejected(self):
        source = ['int F() {\n', 'return 1; }\n']
        notes = [{'path': 'hccl/src/a.cc', 'line': n, 'code': source[n - 1].rstrip('\n'),
                  'note': '函数范围测试。'} for n in (1, 2)]
        data = self.manifest(notes)
        data['functions'][0]['name'] = 'F'
        with patch.object(guide, 'snapshot', return_value=source):
            guide.validate_manifest(data)
            data['functions'].append(dict(data['functions'][0]))
            with self.assertRaisesRegex(ValueError, 'Overlapping function ranges'):
                guide.validate_manifest(data)
            data['functions'].pop()
            data['functions'][0]['name'] = 'NotDefinedHere'
            with self.assertRaisesRegex(ValueError, 'Function signature missing'):
                guide.validate_manifest(data)


if __name__ == '__main__':
    unittest.main()
