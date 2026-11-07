"""Same 1000 questions; expression-derived explanations + supervised tool calls."""
import ast
import copy
import hashlib
import json
import math
from math_tutor.common import ROOT, read_records, write_lines, dump
from math_tutor.data.prepare import expression
from math_tutor.inference.calculator import TOOL_SYSTEM, execute


def relation(equation):
    node=ast.parse(equation,mode='eval').body
    operations={ast.Add:'和',ast.Sub:'差',ast.Mult:'积',ast.Div:'商',ast.Pow:'幂'}
    if isinstance(node,ast.BinOp):
        a,b=ast.unparse(node.left),ast.unparse(node.right)
        return f'所求量按({a})与({b})的{operations[type(node.op)]}建立关系，括号内先计算。'
    return f'所求量由表达式{equation}表示。'


def main():
    out=ROOT/'data/versions/relation-tool-v1'
    if out.exists(): raise SystemExit('Data version already exists')
    manifest={}
    for split in ('train','valid'):
        path=ROOT/f'data/processed/{split}.jsonl'
        rows=read_records(path); converted=[]
        for row in rows:
            eq,value=expression(row['equation'])
            if not math.isclose(value,row['answer'],rel_tol=1e-6,abs_tol=1e-6):
                raise ValueError('Inconsistent source '+row['id'])
            call=json.dumps({'name':'calculator','expression':eq},ensure_ascii=False,separators=(',',':'))
            target=f'关系：{relation(eq)}\n列式：{eq}\n工具调用：{call}'
            if not execute(target)['ok']:raise ValueError('Invalid target')
            item=copy.deepcopy(row)
            item['conversations']=[{'from':'system','value':TOOL_SYSTEM},{'from':'human','value':row['question']},{'from':'gpt','value':target}]
            converted.append(item)
        write_lines(out/f'processed/{split}.jsonl',converted)
        write_lines(out/f'sft_{split}/{split}.jsonl',[{'conversations':r['conversations']} for r in converted])
        manifest[split]={'count':len(rows),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest['limitations']='Deterministic expression-structure explanations, not independently verified semantic quantity explanations. Calculator validates every training target; model learns calls, not arithmetic.'
    dump(out/'manifest.json',manifest)
    config=json.loads((ROOT/'configs/local.json').read_text(encoding='utf-8-sig'))
    config.update(train_file_dir='data/versions/relation-tool-v1/sft_train',validation_file_dir='data/versions/relation-tool-v1/sft_valid',output_dir='runs/relation-tool-v1',model_max_length=384)
    dump(ROOT/'configs/relation-tool.json',config)
    print(json.dumps(manifest,ensure_ascii=False))

if __name__=='__main__':main()
