"""Conservative numeric scoring; never recompute the model's solution."""
import collections
import math
import re
import unicodedata
from fractions import Fraction
VERSION = 'numeric-v2'
ATOM = r'[+\-]?(?:\d+(?:\.\d+)?|\.\d+)(?:\s*/\s*\d+(?:\.\d+)?)?\s*%?'
UNIT = r'(?:[\u4e00-\u9fff]+|[a-zA-Z]+[²³23]?|[/%·^])'

def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('−', '-')
    text = re.sub(r'\\(?:dfrac|tfrac|frac)\s*\{([+\-]?\d+)\}\s*\{(\d+)\}', r'(\1/\2)', text)
    text = re.sub(r'(?<=\d),(?=\d{3}(?:\D|$))', '', text)
    text = re.sub(r'\\(?:left|right|,|!|;|quad|qquad)', '', text)
    for a,b in [('\\%','%'),('\\(',''),('\\)',''),('\\[',''),('\\]',''),('$',''),('**','')]:
        text=text.replace(a,b)
    return text.strip()

def number(s):
    s=s.strip().replace(' ','')
    percent=s.endswith('%')
    if percent: s=s[:-1]
    try:
        value=float(Fraction(s))
        return value/100 if percent else value
    except (ValueError,ZeroDivisionError,OverflowError):
        return None

def numeric_fragment(text):
    s=normalize(text).strip(' \t。.!！,，:：')
    s=re.sub(r'^\(('+ATOM+r')\)',r'\1',s)
    m=re.fullmatch(r'('+ATOM+r')\s*(?:\('+UNIT+r'+\)|'+UNIT+r'+)?',s)
    if not m or re.search(r'或|到|至|以及|和',s): return None
    return number(m.group(1))

def conclusion_value(text):
    s=normalize(text)
    if re.search(r'或|大约|约为|≈|可能',s): return None
    if '=' in s: return numeric_fragment(s.rsplit('=',1)[1])
    tokens=list(re.finditer(ATOM,s))
    return numeric_fragment(s[tokens[0].start():]) if len(tokens)==1 else None

def extract_details(text):
    s=normalize(text)
    candidates=[]
    unresolved=[]
    conclusions=[]
    lines=s.splitlines()
    for line in lines:
        for m in re.finditer(r'(?:答案\s*[:：]|(?<![\u4e00-\u9fff])答\s*[:：]|####)\s*(.*)$',line):
            fragment=m.group(1).strip()
            if fragment in ('','数值','数值。'): continue
            value=numeric_fragment(fragment)
            if value is None and re.match(r'(?:平均|共|一共|还剩|需要)',fragment):
                value=conclusion_value(fragment)
            if value is None: unresolved.append(fragment)
            else: candidates.append(dict(value=value,method='answer_marker',evidence=line))
        if re.match(r'\s*(?:所以|因此|综上|故[，,:：]|由此可知)',line):
            value=conclusion_value(line)
            if value is not None: conclusions.append(dict(value=value,method='conclusion',evidence=line))
    if conclusions: candidates.append(conclusions[-1])
    if unresolved:
        return dict(status='unparsed',prediction=None,candidates=candidates,reason='显式答案无法可靠解析，需人工复核',unresolved=unresolved)
    if not candidates:
        value=numeric_fragment(s)
        if value is not None: candidates.append(dict(value=value,method='bare_number',evidence=s))
    if not candidates:
        lines=[x.strip() for x in lines if x.strip() and not re.fullmatch(r'(?:\d+\.\s*)?答案\s*[:：]\s*数值[。.]?',x.strip())]
        if lines and '=' in lines[-1] and not re.search(r'≈|或|可能',lines[-1]):
            left,right=lines[-1].rsplit('=',1)
            value=numeric_fragment(right)
            if value is not None and re.search(r'[+*/÷×\-]|\b[xX]\b|\d',left):
                candidates.append(dict(value=value,method='final_equation_rhs',evidence=lines[-1]))
    if not candidates: return dict(status='unparsed',prediction=None,candidates=[],reason='没有可可靠提取的最终数值')
    values=[]
    for c in candidates:
        if not any(math.isclose(c['value'],v,rel_tol=1e-9,abs_tol=1e-9) for v in values): values.append(c['value'])
    if len(values)>1: return dict(status='conflict',prediction=None,candidates=candidates,reason='答案或结论互相矛盾，需人工复核')
    return dict(status='parsed',prediction=values[0],candidates=candidates,reason='从明确答案、结论、纯数值或最后等式右端提取')

def score_response(question,response,reference):
    result=extract_details(response)
    result['scorer_version']=VERSION
    if len(re.findall(r'多少|几分之几|百分之几',question))>=2:
        result.update(status='needs_review',correct=None,reason='疑似多问但只有一个数值参考，需审核题目')
    elif result['status']!='parsed': result['correct']=None
    else:
        ok=math.isclose(result['prediction'],float(reference),rel_tol=1e-5,abs_tol=1e-5)
        result.update(status='correct' if ok else 'incorrect',correct=ok)
    return result

def summarize(rows):
    counts=collections.Counter(r['status'] for r in rows)
    n=len(rows)
    judged=counts['correct']+counts['incorrect']
    return dict(scorer_version=VERSION,n=n,status_counts=dict(counts),
        numeric_accuracy=counts['correct']/n if n else None,
        judged_numeric_accuracy=counts['correct']/judged if judged else None,
        judged_count=judged,coverage=judged/n if n else None,
        parse_rate=sum(r.get('prediction') is not None for r in rows)/n if n else None,
        note='numeric_accuracy是全体样本上自动确认正确比例；未解析/冲突/多问不是已确认数学错误。judged_numeric_accuracy仅为已判定子集的辅助指标，必须同时查看coverage。只评价数值，不验证单位、推导或教学质量。')
