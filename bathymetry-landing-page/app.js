const models=[
  {name:'MLP',kind:'Classical',mae:4.9935,rmse:6.3775,r2:.4836},
  {name:'XGBoost',kind:'Classical',mae:4.9535,rmse:6.4577,r2:.4706},
  {name:'Quantum M3',kind:'Quantum',mae:5.1573,rmse:6.5028,r2:.4631},
  {name:'SVR',kind:'Classical',mae:4.7868,rmse:6.6097,r2:.4453},
  {name:'Random Forest',kind:'Classical',mae:5.0608,rmse:6.6561,r2:.4375},
  {name:'Linear Reg.',kind:'Classical',mae:5.3373,rmse:6.6804,r2:.4334},
  {name:'Quantum M2',kind:'Quantum',mae:7.6518,rmse:8.7164,r2:.0354},
  {name:'Quantum M1',kind:'Quantum',mae:12.8481,rmse:15.5530,r2:-2.0711}
];
const chart=document.querySelector('#model-chart');
const tabs=document.querySelectorAll('.metric-tabs button');
function render(metric){
  const lower=metric!=='r2';
  const sorted=[...models].sort((a,b)=>lower?a[metric]-b[metric]:b[metric]-a[metric]);
  const vals=sorted.map(m=>m[metric]); const min=Math.min(...vals),max=Math.max(...vals);
  chart.innerHTML=sorted.map((m,i)=>{
    const normalized=metric==='r2'?Math.max(4,((m[metric]-min)/(max-min))*94+6):Math.max(8,(m[metric]/max)*100);
    return `<div class="bar-row ${i===0?'best':''} ${m.kind==='Quantum'?'quantum-row':''}"><div>${m.name}<span class="bar-kind">${m.kind}</span></div><div class="bar-track"><div class="bar" style="width:${normalized}%"></div></div><div class="bar-value">${m[metric].toFixed(4)}</div></div>`;
  }).join('');
}
tabs.forEach(btn=>btn.addEventListener('click',()=>{tabs.forEach(t=>{t.classList.remove('active');t.setAttribute('aria-selected','false')});btn.classList.add('active');btn.setAttribute('aria-selected','true');render(btn.dataset.metric)}));
render('r2');
