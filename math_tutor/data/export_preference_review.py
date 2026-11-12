"""Build a self-contained, editable human-review page without approving training pairs."""
import argparse
from collections import Counter
import json

from math_tutor.common import ROOT, read_records
from math_tutor.data.preference_io import atomic_json, sha256

PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>数学偏好数据人工复核</title>
<style>body{font:16px/1.6 system-ui,sans-serif;background:#f3f5f8;color:#18212d;max-width:1100px;margin:auto;padding:24px}header{position:sticky;top:0;background:#f3f5f8;padding:12px;z-index:1;border-bottom:1px solid #ccd}section{background:white;border:1px solid #ccd;border-radius:10px;padding:20px;margin:20px 0}.candidate{border-left:4px solid #b9c6d5;padding:12px;margin:14px 0;background:#f8fafc}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:inherit;background:#edf1f6;padding:12px}select,button,textarea{font:inherit;padding:7px;margin:4px}textarea{display:block;width:94%;min-height:55px}button{cursor:pointer}.warn{color:#a34900}.label{font-weight:600}.meta{font-size:14px;color:#516173}</style>
<header><b>50题真实候选人工复核</b><div>先核查题意与参考，再审核候选。教师意见是辅助建议。</div>
<button id="save">导出人工复核 JSON</button><label>导入已保存结果 <input id="load" type="file" accept=".json"></label><span id="progress"></span></header>
<p>每题可选出一个正确回答和一个明确错误回答作为偏好对。全部错误、全部正确、条件不足或有疑问时，可选择跳过或待确认。导出文件保留全部原始候选的关联信息。</p>
<main id="questions"></main>
<script>
const DATA=__DATA__;
const storageKey='math-preference-review-'+DATA.source_sha256;
let edits={};
try{edits=JSON.parse(localStorage.getItem(storageKey)||'{}')}catch(error){}
const questionOptions=[['pending','待复核'],['valid','题目和参考可用'],['invalid','题目或参考有误'],['uncertain','无法确定']];
const candidateOptions=[['pending','待复核'],['correct','正确'],['incorrect','错误'],['uncertain','无法确定']];
function node(tag,text,parent,cls){const element=document.createElement(tag);if(text!==null)element.textContent=text;if(cls)element.className=cls;if(parent)parent.append(element);return element}
function persist(){try{localStorage.setItem(storageKey,JSON.stringify(edits))}catch(error){}const states=Object.values(edits);document.getElementById('progress').textContent=' 已批准 '+states.filter(x=>x.pair_status==='approved').length+' 题 / 已标记 '+states.filter(x=>x.pair_status!=='pending').length+' 题'}
function select(parent,options,value,change){const element=node('select',null,parent);for(const [key,label] of options){const item=node('option',label,element);item.value=key}element.value=value;element.onchange=()=>{change(element.value);persist()};return element}
function render(){document.getElementById('questions').replaceChildren();for(const [index,row] of DATA.records.entries()){
let state=edits[row.id];if(!state){state={question_status:'pending',candidates:{},chosen:'',rejected:'',pair_status:'pending',note:''};edits[row.id]=state}
const section=node('section',null,document.getElementById('questions'));node('h2',(index+1)+'. '+row.id,section);node('p',row.question,section,'label');node('pre','参考列式 '+row.equation+'\n参考答案 '+row.answer,section);
node('p','教师题目判断 '+row.teacher.review.question_status+' / 参考判断 '+row.teacher.review.reference_status+'\n'+row.teacher.review.question_reason,section);
node('span','人工核查题目 ',section);select(section,questionOptions,state.question_status,v=>state.question_status=v);
const options=[['','尚未选择']];for(const candidate of row.candidates){const id=candidate.candidate_id;options.push([id,id]);const box=node('div',null,section,'candidate');node('b',id,box);node('pre',candidate.text,box);
node('p','生成结束 '+candidate.finish_reason+' / '+candidate.generated_tokens+' token',box,'meta');const checks=candidate.automatic;node('p','自动数值检查 '+checks.numeric.status+' / 标记 '+(checks.flags.join(', ')||'无'),box,checks.flags.length?'warn':'meta');
node('pre',JSON.stringify(checks.expressions,null,2),box,'meta');const review=row.teacher.review.candidates.find(x=>x.candidate_id===id);node('p','教师 '+review.verdict+' / '+review.error_type+' / '+review.confidence+'\n'+review.reason,box);
if(!state.candidates[id])state.candidates[id]={verdict:'pending',note:''};const edit=state.candidates[id];node('span','人工候选判断 ',box);select(box,candidateOptions,edit.verdict,v=>edit.verdict=v);const note=node('textarea',null,box);note.placeholder='记录错误位置、对教师意见的修正或不确定原因';note.value=edit.note;note.oninput=()=>{edit.note=note.value;persist()}}
node('div','本题偏好对选择',section,'label');node('span','chosen ',section);select(section,options,state.chosen,v=>state.chosen=v);node('span','rejected ',section);select(section,options,state.rejected,v=>state.rejected=v);
node('span','人工处理 ',section);select(section,[['pending','待复核'],['approved','批准这一对'],['skip','跳过本题'],['uncertain','待确认']],state.pair_status,v=>state.pair_status=v);
const note=node('textarea',null,section);note.placeholder='批准时必填：chosen为什么更好，rejected错在哪里。跳过时可说明理由。';note.value=state.note;note.oninput=()=>{state.note=note.value;persist()};}persist()}
document.getElementById('save').onclick=()=>{for(const row of DATA.records){const state=edits[row.id];if(state.pair_status!=='approved')continue;const chosen=row.candidates.find(c=>c.candidate_id===state.chosen);const rejected=row.candidates.find(c=>c.candidate_id===state.rejected);
if(state.question_status!=='valid'||!chosen||!rejected||chosen.text===rejected.text||chosen.finish_reason==='length'||state.candidates[state.chosen].verdict!=='correct'||state.candidates[state.rejected].verdict!=='incorrect'||!state.note.trim()){alert(row.id+' 的批准信息不完整。请确认题目有效、chosen正确且完整、rejected错误、两者不同，并填写理由。');return}}
const result={schema_version:1,source_sha256:DATA.source_sha256,exported_at:new Date().toISOString(),records:DATA.records.map(row=>({question_id:row.id,human:edits[row.id]}))};const url=URL.createObjectURL(new Blob([JSON.stringify(result,null,2)],{type:'application/json'}));const link=node('a',null,null);link.href=url;link.download='manual-review.json';link.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
document.getElementById('load').onchange=async event=>{try{const result=JSON.parse(await event.target.files[0].text());if(result.source_sha256!==DATA.source_sha256)throw Error('数据版本不一致');if(result.records.length!==DATA.records.length)throw Error('题目数量不同');const ids=new Set(DATA.records.map(r=>r.id));if(result.records.some(r=>!ids.has(r.question_id))||new Set(result.records.map(r=>r.question_id)).size!==ids.size)throw Error('题目ID不匹配');edits=Object.fromEntries(result.records.map(r=>[r.question_id,r.human]));render()}catch(error){alert('导入失败 '+error.message)}};
render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', default='data/versions/preference-sample-v1')
    args = parser.parse_args()
    directory = ROOT / args.input
    teachers = {r['id']:r for r in read_records(directory/'teacher_reviews.jsonl')}
    records = read_records(directory/'checked.jsonl')
    summary = Counter()
    for row in records:
        row['teacher'] = teachers[row['id']]
        verdicts = {v['candidate_id']:v for v in row['teacher']['review']['candidates']}
        for candidate in row['candidates']:
            automatic = candidate['automatic']
            teacher = verdicts[candidate['candidate_id']]
            summary[teacher['verdict']] += 1
            if (automatic['numeric']['correct'] is True and teacher['verdict']=='incorrect') or (
                automatic['numeric']['correct'] is False and teacher['verdict']=='correct'):
                automatic['flags'].append('teacher_numeric_disagreement')
                summary['teacher_numeric_disagreements'] += 1
    source_hash = sha256(directory/'checked.jsonl')
    data = dict(source_sha256=source_hash, records=records)
    atomic_json(directory/'review_bundle.json', data)
    encoded = json.dumps(data,ensure_ascii=True).replace('<', chr(92)+'u003c')
    (directory/'人工复核.html').write_text(PAGE.replace('__DATA__',encoded),encoding='utf-8')
    atomic_json(directory/'review_summary.json',dict(questions=len(records),teacher_verdicts=dict(summary),
        human_approved=0, note='Human decisions remain pending until an exported manual review is supplied.'))
    print(dict(questions=len(records),teacher_verdicts=dict(summary)))


if __name__ == '__main__':
    main()
