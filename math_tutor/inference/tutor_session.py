"""Route each tutor turn and verify work using deterministic evidence."""
from dataclasses import dataclass
from functools import lru_cache
import math
import random
import re
import unicodedata

from math_tutor.common import ROOT, read_records
from math_tutor.data.prepare import expression
from math_tutor.inference.calculator import execute
from math_tutor.evaluation.scoring import extract_details, numeric_fragment


TUTOR_SYSTEM = ('你是中文数学辅导老师。根据当前题目和上下文回答，用户只要提示时给一条可执行的提示。'
                '说明关键概念或数量关系。信息不足先询问，避免把未核验的推断说成确定的对错。')

_PRACTICE_HEADER = '练习题：'
_PRACTICE_FOOTER = '\n\n请发你的答案或解题步骤。也可以说“给我一点提示”。'
_FIELD = re.compile(r'(?:^|[\s；;。])\s*(?:我的|学生的?)?(题目|问题|列式|算式|步骤|答案|作答)\s*[:：]\s*')
_MATH = r'[0-9.()+*/%÷×^−\-\s]+'
_WORD = re.compile(r'(?:小明)?有(\d{1,3})盒(?:铅笔|彩笔)[，,]每盒(\d{1,3})支[，,](?:一共|总共)有?多少支[？?]?')
_PRACTICE = re.compile(r'出(?:一|1|个|道|题|点|些|几|\d)|来(?:一|1|个|道|点|些|几|\d).{0,12}题|考考我|给我(?:一|1|几|个|道).{0,14}题|(?:我想|我要|想)练习|给我.{0,8}练习题|再来(?:一|1|个|道)|下一题')
_ASK_CHECK = re.compile(r'对不对|对吗|正确吗|算对了吗|算错了吗|哪里错|哪一步错|错在哪|批改|帮我(?:检查|核对|验算)|检查(?:答案|列式|作答|步骤)')
_HINT = re.compile(r'提示|思路|怎么想|从哪(?:里)?入手|不知道怎么|先别(?:给|说)答案|不(?:要|用)给答案')
_REVEAL = re.compile(r'^(?:请|直接|可以)?(?:给我|告诉我|公布|看看|查看)?(?:正确|完整|最终)?(?:答案|解答|解析)[。！!？?]?$')
_FOLLOW = re.compile(r'^(?:为什么|怎么得(?:出|到)|哪一步|哪里错|再解释|怎么列式)')
_SOLVE = re.compile(r'多少|百分之几|几分之几|还剩几个|怎么解|怎么列式|求(?:出|解|值|面积|周长)|解(?:答|一下|这道题)|算(?:一下|出|算)|等于几')
_CONCEPT = re.compile(r'什么意思|是什么意思|什么是|有什么区别|怎么理解|为什么要')
_RETRY = re.compile(r'^(?:我想|我要)?(?:重做|再试一次|再做一遍|重复题目|题目是什么)[。！!？?]?$')


@dataclass(frozen=True)
class Reference:
    formula: str
    value: float
    source: str


@dataclass(frozen=True)
class Submission:
    question: str = ''
    formula: str = ''
    steps: str = ''
    answer: str = ''

    @property
    def has_work(self):
        return bool(self.formula or self.steps or self.answer)


@dataclass(frozen=True)
class Turn:
    mode: str
    reply: str | None = None


def _text(text):
    return unicodedata.normalize('NFKC', text).replace('−', '-').strip()


def _value(text):
    try:
        return expression(_text(text))[1]
    except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
        return None


def _same(a, b):
    return math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-5)


def _format(number):
    return format(number, '.10g')


def _arithmetic(question):
    text = re.sub(r'^(?:(?:请)?(?:帮我)?(?:计算|算一下|算一算|求值)\s*[:：]?\s*)', '', _text(question))
    text = re.sub(r'\s*(?:的值是多少|等于多少|等于几|是多少)\s*[?？。]*$', '', text)
    text = re.sub(r'\s*(?:=\s*)?[?？。]+$|\s*=\s*$', '', text).strip()
    if re.fullmatch(_MATH, text) and re.search(r'[+*/÷×^-]', text) and re.search(r'\d', text):
        return text
    return None


def _question_key(text):
    return re.sub(r'\s+', '', _text(text)).rstrip('?.。')


@lru_cache(maxsize=1)
def _training_references():
    path = ROOT / 'data' / 'processed' / 'train.jsonl'
    references = {}
    conflicts = set()
    if path.is_file():
        for row in read_records(path):
            key = _question_key(row['question'])
            value = _value(row['equation'])
            if value is not None and _same(value, row['answer']):
                if key in references and not _same(value, references[key].value):
                    conflicts.add(key)
                references[key] = Reference(row['equation'], value, '训练题库的参考算式，经计算器复算')
    return {key: ref for key, ref in references.items() if key not in conflicts}


def reference_for(question):
    question = re.sub(r'^(?:请)?(?:帮我检查|帮我核对|帮我验算|请批改|检查|批改)\s*:?\s*', '', _text(question))
    formula = _arithmetic(question)
    if formula is None:
        match = _WORD.fullmatch(_text(question))
        if match:
            formula = f'{match[1]}×{match[2]}'
    if formula is not None:
        value = _value(formula)
        if value is not None:
            return Reference(formula, value, '题目规则与计算器复算')
    return _training_references().get(_question_key(question))


def parse_submission(text):
    text = _text(text)
    matches = list(_FIELD.finditer(text))
    if matches:
        fields = {}
        aliases = {'题目': 'question', '问题': 'question', '列式': 'formula', '算式': 'formula',
                   '步骤': 'steps', '答案': 'answer', '作答': 'answer'}
        prefix = _ASK_CHECK.sub('', text[:matches[0].start()].strip()).strip(' :，,')
        if prefix and re.search(r'\d|[?？]|多少|几', prefix):
            fields['question'] = prefix
        for i, match in enumerate(matches):
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            value = text[match.end():end].strip()
            field = aliases[match[1]]
            if field == 'answer':
                value = _ASK_CHECK.split(value, maxsplit=1)[0].strip(' ，,。!?')
            fields[field] = '\n'.join(filter(None, (fields.get(field), value)))
        return Submission(**fields)

    # Accept common unlabelled forms, while preserving ambiguous text for chat.
    equation = _ASK_CHECK.split(text, maxsplit=1)[0].strip(' ，,。!?')
    equal = re.fullmatch(rf'\s*({_MATH})\s*=\s*([^=]+)', equation)
    if equal and re.search(r'\d', equal[1]) and numeric_fragment(equal[2]) is not None:
        return Submission(formula=equal[1].strip(), answer=equal[2].strip())
    natural = re.search(r'(?:^|[，,。!?；;\n]\s*)(?:我的答案是|答案是|结果是|应该是|我答的是|我答|我算出来是|我算的是|我算出|我算是|我算|我觉得是|我认为是)\s*(.+)', text, re.S)
    if natural:
        question = text[:natural.start()].strip()
        answer = _ASK_CHECK.split(natural[1], maxsplit=1)[0].strip(' ，,。!?')
        if numeric_fragment(answer) is not None:
            return Submission(question=question, answer=answer)
    equations = list(_equations(text))
    if len(equations) > 1 and all(re.fullmatch(_MATH, side) for _, left, right in equations for side in (left, right)):
        answer = equations[-1][2]
        return Submission(steps=text, answer=answer if numeric_fragment(answer) is not None else '')
    return Submission(answer=text) if numeric_fragment(text) is not None else Submission()


def _equations(steps):
    for index, line in enumerate(re.split(r'[\n;；]+', _text(steps)), 1):
        line = re.sub(r'^(?:第\d+步\s*[:：]\s*|\d+[、.)]\s+)', '', line.strip())
        if '=' not in line:
            continue
        parts = [part.strip() for part in line.split('=')]
        if parts[0].lower() == 'x':
            parts.pop(0)
        for left, right in zip(parts, parts[1:]):
            yield index, left, right


def grade(submission, reference, hint_only=False):
    """Report independently verified arithmetic, reference match, and scope."""
    formula = submission.formula
    formula_value = _value(formula) if formula else None
    parsed = extract_details(submission.answer) if submission.answer else None
    answer = parsed['prediction'] if parsed else None
    notes = []
    first_error = None
    if formula:
        if formula_value is None:
            notes.append('列式暂不可复算，请用一条完整的四则算式。')
        elif reference and not _same(formula_value, reference.value):
            first_error = '首个可确认的问题在列式：复算结果与参考数值不一致。'
            notes.append(first_error)
        elif reference:
            notes.append('列式复算结果与参考数值一致。')
        else:
            notes.append('已复算你的列式，列式是否符合题意仍待核验。' if hint_only else
                         f'你的列式复算得 {_format(formula_value)}，列式是否符合题意仍待核验。')

    checked, unknown = 0, 0
    for index, left, right in _equations(submission.steps):
        left_value, right_value = _value(left), _value(right)
        if left_value is None or right_value is None:
            unknown += 1
            continue
        checked += 1
        if not _same(left_value, right_value):
            if hint_only:
                step_error = f'第 {index} 行等式两侧的值不同，先重算这一行。'
            else:
                step_error = f'第 {index} 行有计算错误：{left} = {_format(left_value)}，你写的是 {right}。'
            notes.append(step_error)
            first_error = first_error or step_error
            break
    if submission.steps:
        if not checked or unknown:
            notes.append('部分步骤待核验。可以把每一步改写成一行数值等式。')
        elif not first_error:
            notes.append(f'已核验 {checked} 处等式，数值计算一致。')

    if answer is not None:
        if formula_value is not None:
            if _same(formula_value, answer):
                notes.append('你的答案与所列算式的计算结果一致。')
            else:
                notes.append('计算有误：你的答案和所列算式的结果不一致。'
                             if hint_only else
                             f'计算有误：所列算式算得 {_format(formula_value)}，与提交的答案不同。')
        if reference:
            status = '最终数值正确。' if _same(answer, reference.value) else '最终数值有误。'
            notes.insert(0, status)
    elif submission.answer:
        notes.append('答案待核验，请提供一个明确的数字、分数或百分数。')

    if reference and not hint_only:
        notes.append(f'参考结果：{reference.formula} = {_format(reference.value)}。依据：{reference.source}。')
    elif reference and hint_only:
        notes.append(_hint(reference))
    elif not reference and (submission.question or not (formula_value is not None or checked)):
        notes.append('题意或参考答案待核验。请补充完整题目及列式；我可以继续检查每一步计算。')
    return '\n'.join(notes)


def _hint(reference):
    if reference and reference.source.startswith('题目规则'):
        formula = reference.formula
        if '%' in formula:
            return '提示：先把百分数换成分数或小数。'
        if '(' in formula or '（' in formula:
            return '提示：先算括号里的部分。'
        if '×' in formula:
            return '提示：如果是每组同样多，试试用组数乘每组数量。'
        if '/' in formula and '+' in formula:
            return '提示：试试先通分，再把分子相加。'
    return '提示：先明确题目求什么，再逐步核对运算和每一步的结果。'


def _topic(text):
    for topic, pattern in (
        ('应用题', '应用|文字|盒'), ('百分数', '百分'), ('分数', '分数'),
        ('除法', '除法|除法题|除以'), ('乘法', '乘法|乘以'),
        ('减法', '减法'), ('加法', '加法'), ('混合运算', '混合|四则'),
    ):
        if re.search(pattern, text):
            return topic
    if re.fullmatch(r'[\d.()+*/%÷×^= ?？\-]+', text.strip()):
        for symbol, topic in (('%', '百分数'), ('(', '混合运算'), ('/', '分数'), ('÷', '除法'),
                              ('×', '乘法'), ('-', '减法'), ('+', '加法')):
            if symbol in text:
                return topic
    return None


def _new_practice(text, previous=''):
    if re.search(r'几何|证明|微积分|方程|概率|函数|矩阵|根号', text):
        return '目前可以出四则运算、分数、百分数和每组同样多的应用题。你想练哪一种？'
    topic = _topic(text) or _topic(previous) or '混合运算'
    difficult = bool(re.search(r'难|进阶|复杂', text))
    a, b = [random.randint(12, 28) if difficult else random.randint(2, 9) for _ in range(2)]
    questions = {
        '应用题': f'有{a}盒铅笔，每盒{b}支，一共有多少支？',
        '百分数': f'{a * 10} × {b * 5}%',
        '分数': f'{a}/30 + {b}/45',
        '除法': f'{a * b} ÷ {b}',
        '乘法': f'{a} × {b}',
        '减法': f'{a + b} - {b}',
        '加法': f'{a} + {b}',
        '混合运算': f'({a} + {b}) × {a if difficult else 3}',
    }
    question = questions[topic]
    if topic != '应用题':
        question += ' = ?'
    if question == previous:
        return _new_practice(text, previous)
    return _PRACTICE_HEADER + question + _PRACTICE_FOOTER


def _looks_like_problem(question):
    if _arithmetic(question) or _WORD.fullmatch(_text(question)):
        return True
    if not _SOLVE.search(question) or _CONCEPT.search(question):
        return False
    return bool(re.search(r'\d|题|算式|方程|函数|面积|周长|分数|角度|概率|算术|百分数', question))


def _context(history):
    """Recover an exercise or a submitted question from the visible transcript."""
    for message in reversed(history):
        content = message['content']
        if message['role'] == 'assistant' and content.startswith(_PRACTICE_HEADER) and content.endswith(_PRACTICE_FOOTER):
            question = content[len(_PRACTICE_HEADER):-len(_PRACTICE_FOOTER)]
            if reference_for(question):
                return question, True
        if message['role'] == 'user':
            submission = parse_submission(content)
            if submission.question:
                return submission.question, False
            if _looks_like_problem(content):
                return content, False
    return '', False


def _with_question(submission, question):
    return Submission(submission.question or question, submission.formula, submission.steps, submission.answer)


def route_turn(messages):
    text = _text(messages[-1]['content'])
    history = messages[:-1]
    submission = parse_submission(text)
    if _PRACTICE.search(text) and not submission.has_work and not submission.question:
        prior, _ = _context(history)
        return Turn('practice', _new_practice(text, prior))
    if submission.question and not submission.has_work and _ASK_CHECK.search(text):
        return Turn('check', '请发你的答案。也可以一起发列式或逐步计算，我会分别核验。')
    if submission.has_work and (submission.question or submission.formula or submission.steps or
                                _ASK_CHECK.search(text) or history):
        prior, _ = _context(history)
        current = _with_question(submission, prior)
        reference = reference_for(current.question) if current.question else None
        if current.question or current.formula or current.steps:
            return Turn('check', grade(current, reference, bool(_HINT.search(text))))
    if not submission.has_work and re.fullmatch(_MATH, text) and _arithmetic(text):
        question, active = _context(history)
        if active:
            return Turn('check', grade(Submission(question=question, formula=text), reference_for(question)))
    if _ASK_CHECK.search(text) or (submission.answer and _FIELD.search(text)):
        return Turn('check', '请把题目和你的答案一起发给我。如果你写了列式或步骤，也可以一并贴上。')

    if _HINT.search(text) or _REVEAL.fullmatch(text) or _RETRY.fullmatch(text) or _FOLLOW.search(text):
        question, active = _context(history)
        reference = reference_for(question) if question else None
        if active and _RETRY.fullmatch(text):
            return Turn('practice', _PRACTICE_HEADER + question + _PRACTICE_FOOTER)
        if active and (_HINT.search(text) or (_FOLLOW.search(text) and history[-1]['content'].endswith(_PRACTICE_FOOTER))):
            return Turn('tutor', _hint(reference))
        if reference and (_REVEAL.fullmatch(text) or (active and _FOLLOW.search(text))):
            return Turn('tutor', f'参考算式：{reference.formula}\n答案：{_format(reference.value)}。依据：{reference.source}。')
        return Turn('tutor')

    question = submission.question or text
    if _CONCEPT.search(question):
        return Turn('chat')
    if _looks_like_problem(question):
        reference = reference_for(question)
        if reference:
            return Turn('solve', f'列式：{reference.formula}\n答案：{_format(reference.value)}')
        return Turn('solve')
    return Turn('chat')


def finish_solution(raw):
    result = execute(raw, require_call=False)
    if result['ok']:
        return result['response'] + '\n（计算器已复算；列式请结合题意核对。）'
    return '本次没有可复算的单一算式。以下是模型的解题草稿：\n' + raw
