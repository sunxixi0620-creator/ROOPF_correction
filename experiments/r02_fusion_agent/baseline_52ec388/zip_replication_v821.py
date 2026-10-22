"""原版复现的紧凑支付与恢复执行器；逐函数持久化，恢复不重新调用已付目标。"""
import csv
import gzip
import hashlib
import io
import json
import pickle
import sqlite3
import zlib
from pathlib import Path
import numpy as np
import torch
from scripts.zip_replication_v820 import ROOT, ARMS, build, problem, release

RUN=ROOT/'results/zip_replication_v821'
OUT=ROOT/'docs/revision/zip_replication820_results'
TASKS=[(g,f) for g,n in [('BBOB',24),('CEC-style',6),('UAV',5)] for f in range(1,n+1)]


def pack(x):
    """只序列化本执行器产生的字节与数值，不保存每步模型副本。"""
    return zlib.compress(pickle.dumps(x,protocol=5),3)


def unpack(x):
    """只读取本项目带冻结身份的内部支付记录。"""
    return pickle.loads(zlib.decompress(x))


def raw(x):
    """CPU float32原始字节用于精确身份，禁止近似点缓存。"""
    assert x.dtype==torch.float32 and x.device.type=='cpu'
    return x.detach().contiguous().numpy().tobytes()


def digest(x):
    """内容身份用于检测状态或响应被替换。"""
    return hashlib.sha256(x).hexdigest()


def write(path,data):
    """已保存科学证据必须逐字一致，不覆盖旧结果。"""
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): assert path.read_bytes()==data,str(path)
    else:
        pending=path.with_suffix(path.suffix+'.pending');pending.write_bytes(data);pending.replace(path)


class Paid:
    """先保存调用意图再支付；未知响应禁止自动重付，首批标签在六组间共享。"""
    def __init__(self,p,db,job,guard,verify,stochastic):
        self.p,self.db,self.job,self.guard,self.verify=p,db,job,guard,verify
        self.fun,self.dim=p.fun,10
        self.index=0;self.values=[];self.stochastic=stochastic
    def repaire(self,x):
        """沿用发布版边界修复。"""
        return self.p.repaire(x)
    def getfunname(self):
        """保持模型函数类型分支的原输入名称。"""
        return self.p.getfunname()
    def calfitness(self,x):
        """完整批输入缓存避免改变BBOB批函数语义；随机f20只复用同一支付事务。"""
        self.guard.check();idx=self.index;self.index+=1
        job=self.job if idx else self.job.split(':')[0]+':initial'
        xb=raw(x);rng=torch.get_rng_state();before=rng.numpy().tobytes()
        request=pack(dict(x=xb,shape=tuple(x.shape),rng=digest(before)))
        row=self.db.execute('select request,response from calls where job=? and idx=?',(job,idx)).fetchone()
        if row:
            assert row[0]==request and row[1] is not None,'调用身份变化或支付响应缺失'
            response=unpack(row[1])
        else:
            assert not self.verify,'恢复出现尚未支付请求'
            key=digest(xb+repr(tuple(x.shape)).encode())
            cached=None if self.stochastic else self.db.execute('select response from cache where key=?',(key,)).fetchone()
            n=0 if cached else x.numel()//10
            used=self.db.execute('select coalesce(sum(n),0) from calls').fetchone()[0]
            assert used+n<=130000
            self.db.execute('insert into calls values(?,?,?,?,?)',(job,idx,request,None,n));self.db.commit()
            if cached: response=unpack(cached[0])
            else:
                y=self.p.calfitness(x).detach().reshape(x.shape[:2]).float()
                assert torch.isfinite(y).all()
                after=torch.get_rng_state().numpy().tobytes()
                response=dict(y=raw(y),shape=tuple(y.shape),rng_after=after if after!=before else None)
                if not self.stochastic:
                    assert response['rng_after'] is None,'声明确定性的函数消耗了随机流'
                    self.db.execute('insert into cache values(?,?)',(key,pack(response)))
            self.db.execute('update calls set response=? where job=? and idx=?',(pack(response),job,idx));self.db.commit()
        if response['rng_after'] is not None:
            torch.set_rng_state(torch.from_numpy(np.frombuffer(response['rng_after'],dtype=np.uint8).copy()))
        y=torch.from_numpy(np.frombuffer(response['y'],dtype=np.float32).copy()).reshape(response['shape'])
        self.values.append(y);return y


def run_task(guard,task,verify=False,first=False):
    """full连续随机流沿发布入口推进；其他组共享初始点且不能扰动下一批full随机流。"""
    auth=json.loads((RUN/'AUTHORIZATION.json').read_text())
    assert auth['approved'] and auth['max_new_native_FE']==4550000
    binding=json.loads((ROOT/'docs/revision/zip_replication820/BINDING.json').read_text())
    for name,sha in binding.items(): assert digest((ROOT/name).read_bytes())==sha,name
    group,fid=TASKS[task];folder=RUN/f'{group}_f{fid:02d}';folder.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect('file:'+str(folder/'payments.sqlite')+('?mode=ro' if verify else '?mode=rwc'),uri=True)
    if not verify:
        db.executescript('pragma synchronous=FULL; CREATE TABLE IF NOT EXISTS calls(job TEXT,idx INTEGER,request BLOB,response BLOB,n INTEGER,PRIMARY KEY(job,idx)); CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,response BLOB);')
    assert db.execute('pragma integrity_check').fetchone()[0]=='ok'
    assert db.execute('select count(*) from calls where response is null').fetchone()[0]==0
    release.set_seed(20260630);models={'full':build('full')}
    for arm in ARMS[1:]:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20260630);models[arm]=build(arm)
    # 模型的跨批动态属性仅为输出日志；参数和随机特征不得在搜索中改写。
    initial_states={a:{k:v.clone() for k,v in m.state_dict().items()} for a,m in models.items()}
    if fid>1:
        previous=RUN/f'{group}_f{fid-1:02d}'/'FULL_RNG.bin'
        torch.set_rng_state(torch.from_numpy(np.frombuffer(previous.read_bytes(),dtype=np.uint8).copy()))
    p=problem(group,fid);records={a:[] for a in ARMS}
    for batch in range(1 if first else 10):
        guard.check();x=p.genRandomPop((10,100,10));start=torch.get_rng_state().clone()
        full_after=None
        for arm in ARMS:
            torch.set_rng_state(start)
            paid=Paid(p,db,f'b{batch:02d}:{arm}',guard,verify,group=='BBOB' and fid==20)
            with torch.no_grad(): _,trail,nfe,_=models[arm](x.clone(),paid)
            values=torch.cat(paid.values,1);assert nfe==300 and values.shape==(10,300)
            assert torch.equal(values.min(1).values,trail[:,-1])
            for k,v in models[arm].state_dict().items(): assert torch.equal(v,initial_states[arm][k]),(arm,k)
            records[arm].append(values.numpy());data=pack(dict(values=raw(values),rng=torch.get_rng_state().numpy().tobytes()))
            target=folder/f'b{batch:02d}_{arm}.bin'
            if verify: assert target.read_bytes()==data
            else: write(target,data)
            if arm=='full': full_after=torch.get_rng_state().clone()
        torch.set_rng_state(full_after)
        print(json.dumps(dict(task=task,batch=batch,verify=verify,native_FE=db.execute('select sum(n) from calls').fetchone()[0])),flush=True)
    if not first:
        write(folder/'FULL_RNG.bin',torch.get_rng_state().numpy().tobytes())
        summary(guard,folder,records,group,fid,db,verify)
    db.close()


def summary(guard,folder,records,group,fid,db,verify):
    """逐函数保存原始值、配对消融和原表排名；全部负结果同样交付。"""
    dest=OUT/f'{group}_f{fid:02d}'
    if verify:
        assert all((dest/n).is_file() for n in ('FUNCTION_RESULTS.csv','RANKS.csv','COMPARISONS.csv','ABLATIONS.csv','VALUES.bin','FEES.json','FULL_RNG.bin','README.zh-CN.md','PAYMENTS.sqlite.gz'))
    dest.mkdir(parents=True,exist_ok=True)
    vals={a:np.concatenate(records[a]) for a in ARMS}
    finals={a:vals[a].min(axis=1).astype(np.float64) for a in ARMS}
    rows=[dict(group=group,fid=fid,arm=a,n=100,mean=float(finals[a].mean()),std=float(finals[a].std())) for a in ARMS]
    base=[b for b in csv.DictReader((ROOT/'docs/revision/repaired_paper641/BASELINES.csv').open()) if b['group']==group and int(b['fid'])==fid]
    means={'full':rows[0]['mean'],**{b['method']:float(b['mean']) for b in base}}
    ranks=[dict(method=m,rank=1+sum(v<x for v in means.values())+.5*(sum(v==x for v in means.values())-1)) for m,x in means.items()]
    comparisons=[dict(method=b['method'],full_mean=means['full'],baseline_mean=float(b['mean']),baseline_std=float(b['std']),W=int(means['full']<float(b['mean'])),T=int(means['full']==float(b['mean'])),L=int(means['full']>float(b['mean']))) for b in base]
    ablations=[]
    for a in ARMS[1:]:
        d=finals['full']-finals[a]
        ablations.append(dict(arm=a,mean_difference=float(d.mean()),W=int(d.mean()<0),T=int(d.mean()==0),L=int(d.mean()>0),paired_W=int((d<0).sum()),paired_T=int((d==0).sum()),paired_L=int((d>0).sum())))
    for name,table in [('FUNCTION_RESULTS.csv',rows),('RANKS.csv',ranks),('COMPARISONS.csv',comparisons),('ABLATIONS.csv',ablations)]:
        stream=io.StringIO(newline='');writer=csv.DictWriter(stream,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
        write(dest/name,stream.getvalue().encode())
    # 压缩原始300个评价值可重算逐轨迹终值及全程最好曲线；不只保存均值。
    write(dest/'VALUES.bin',pack({a:(v.shape,v.tobytes()) for a,v in vals.items()}))
    fees=dict(native_FE=db.execute('select sum(n) from calls').fetchone()[0],logical_FE=180000,baseline_FE=0,training=0)
    write(dest/'FEES.json',(json.dumps(fees,indent=2)+'\n').encode())
    write(dest/'FULL_RNG.bin',(folder/'FULL_RNG.bin').read_bytes())
    note='原表基线不是同初始配对；CEC严格采用发布入口，原表偏移身份未确认；BBOB-f20单列参考。训练集消融尚缺权重身份，不计为完成。'
    write(dest/'README.zh-CN.md',('# 原发布版 '+group+' f'+str(fid)+'\n\n'+note+'\n\n目标值、平均排名越小越好。均值及总体标准差见FUNCTION_RESULTS.csv，完整模块消融见ABLATIONS.csv。\n').encode())
    if not verify:
        # 支付库必须随每函数远端交付，避免仅保存哈希导致再度丢失唯一证据。
        db.commit();write(dest/'PAYMENTS.sqlite.gz',gzip.compress((folder/'payments.sqlite').read_bytes(),mtime=0))
    else:
        assert gzip.decompress((dest/'PAYMENTS.sqlite.gz').read_bytes())==(folder/'payments.sqlite').read_bytes()
    assert sum(p.stat().st_size for p in dest.iterdir())<=32*1024**2
    assert all(p.stat().st_size<=16*1024**2 for p in dest.iterdir())
    guard.check()


def contract(guard,verify=False):
    """合成目标不计真实实验FE；验证入口数值等价、整模块禁用和支付日志只读重放。"""
    from types import SimpleNamespace
    class Toy:
        """只用于合同的确定性合成函数，不提供任何论文成绩。"""
        dim=10
        fun={'fid':1,'xlb':-5,'xub':5}
        def repaire(self,x):
            """固定边界用于验证发布输入接口。"""
            return x.clamp(-5,5)
        def getfunname(self):
            """模拟普通BBOB分支。"""
            return 1
        def calfitness(self,x):
            """合成平方和只检测执行与恢复，不评估模型质量。"""
            return (x*x).sum(-1)
    p=Toy();folder=RUN/'contract';folder.mkdir(parents=True,exist_ok=True)
    expected={}
    for arm in ARMS:
        guard.check();release.set_seed(20260630);m=build(arm);m.MaxNFE=120
        torch.manual_seed(820);x=torch.rand(10,100,10)*10-5
        with torch.no_grad(): _,trail,nfe,_=m(x,p)
        assert nfe==120 and torch.isfinite(trail).all()
        expected[arm]=digest(raw(trail))
        if arm=='full':
            release.set_seed(20260630);literal=release.build_optimizer(SimpleNamespace(dimension=10,population_size=100,budget=120))
            torch.manual_seed(820);xx=torch.rand(10,100,10)*10-5
            with torch.no_grad(): _,other,_,_=literal(xx,p)
            assert torch.equal(trail,other)
        if arm=='anchor_only': assert not hasattr(m,'surrogate') and not hasattr(m,'router_model')
        if arm=='online_only': assert m.baseline_model is None and m.router_model is None
    db=sqlite3.connect(folder/'synthetic.sqlite')
    db.executescript('CREATE TABLE IF NOT EXISTS calls(job TEXT,idx INTEGER,request BLOB,response BLOB,n INTEGER,PRIMARY KEY(job,idx)); CREATE TABLE IF NOT EXISTS cache(key TEXT PRIMARY KEY,response BLOB);')
    for readonly in (False,True):
        release.set_seed(20260630);m=build('full');m.MaxNFE=120
        torch.manual_seed(820);x=torch.rand(10,100,10)*10-5
        paid=Paid(p,db,'b00:full',guard,readonly,False)
        with torch.no_grad(): _,trail,_,_=m(x,paid)
        assert digest(raw(trail))==expected['full']
    assert db.execute('select sum(n) from calls').fetchone()[0]<=1200
    db.close()
    data=(json.dumps(dict(status='passed',real_FE=0,training=0,synthetic_only=True,hashes=expected),sort_keys=True,indent=2)+'\n').encode()
    write(folder/'CONTRACT.json',data)
