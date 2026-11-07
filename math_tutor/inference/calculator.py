"""Deterministic calculator protocol: no eval, no reference-answer access."""
import json
import re
import unicodedata
from math_tutor.data.prepare import expression

TOOL_SYSTEM = '你是数学助手。先用一句简短的关系解释说明列式，再输出列式和计算器调用。只输出三行：关系：解释；列式：数学表达式；工具调用：{"name":"calculator","expression":"同一个数学表达式"}。不要自己计算或输出答案，答案由计算器返回。'


def execute(response, require_call=True):
    text=unicodedata.normalize('NFKC',response).replace('−','-')
    formulas=re.findall(r'^\s*列式\s*:\s*(.*?)\s*$',text,re.M)
    calls=re.findall(r'^\s*工具调用\s*:\s*(.*?)\s*$',text,re.M)
    result={'ok':False,'raw_response':response}
    if len(formulas)!=1:
        return dict(result,error='Expected exactly one formula')
    try:
        formula,value=expression(formulas[0])
        if require_call:
            if len(calls)!=1: raise ValueError('Expected exactly one calculator call')
            call=json.loads(calls[0])
            if set(call)!={'name','expression'} or call['name']!='calculator' or not isinstance(call['expression'],str):
                raise ValueError('Invalid calculator schema')
            called,_=expression(call['expression'])
            if called!=formula: raise ValueError('Formula/call mismatch')
    except (ValueError,TypeError,KeyError,SyntaxError,ZeroDivisionError,OverflowError) as exc:
        return dict(result,error=str(exc))
    relations=re.findall(r'^\s*(关系|分析)\s*:\s*(.*?)\s*$',text,re.M)
    answer=format(value,'.10g')
    rendered=(f'{relations[0][0]}：{relations[0][1]}\n' if len(relations)==1 else '')+f'列式：{formula}\n答案：{answer}'
    return dict(result,ok=True,expression=formula,value=value,answer=answer,response=rendered,
                relation_present=len(relations)==1)
