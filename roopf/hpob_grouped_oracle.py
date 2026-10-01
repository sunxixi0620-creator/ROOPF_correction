"""Separate evaluator process: public X, durable paid calls, post-run metrics."""
import json
import os
import multiprocessing as mp
from pathlib import Path
import numpy as np
from roopf.experiment_io import atomic_path


def serve(connection,label_file,journal_file,budget):
    try:
        y=np.load(label_file,allow_pickle=False).reshape(-1)
        path=Path(journal_file);path.parent.mkdir(parents=True,exist_ok=True)
        # Atomic whole-journal replacement; no partial final line after a crash.
        records=json.loads(path.read_text()) if path.exists() else []
        assert len(records)<=budget and len({r['index'] for r in records})==len(records)
        assert all(r['value']==float(y[r['index']]) for r in records)
        connection.send(dict(paid=records))
        while True:
            cmd=connection.recv()
            if cmd['op']=='close':break
            try:
                if cmd['op']=='query':
                    index=cmd['index']
                    if type(index) is not int or not 0<=index<len(y):raise ValueError('Invalid index')
                    if len(records)>=budget:raise ValueError('Budget exhausted')
                    if index in {r['index'] for r in records}:raise ValueError('Duplicate query')
                    record=dict(index=index,value=float(y[index]),sequence=len(records))
                    records.append(record)
                    with atomic_path(path) as p:p.write_text(json.dumps(records,allow_nan=False))
                    connection.send(dict(record=record))
                elif cmd['op']=='metrics':
                    if len(records)!=budget:raise ValueError('Metrics withheld until budget completion')
                    values=np.array([r['value'] for r in records]);lo=float(y.min());hi=float(y.max())
                    regret=(hi-np.maximum.accumulate(values))/(hi-lo) if hi>lo else np.zeros(budget)
                    hits=np.flatnonzero(regret<=.05)
                    connection.send(dict(metrics=dict(regret=regret.tolist(),
                        cold_auc=float(regret[4:40].mean()),auc=float(regret[4:].mean()),
                        terminal=float(regret[-1]),time=int(hits[0]+1) if len(hits) else budget+1,
                        # Report time starts with the full common 5-observation design.
                        restricted_time=max(5,int(hits[0]+1)) if len(hits) else budget+1,
                        attained=bool(len(hits)),constant_table=hi==lo)))
                else:raise ValueError('Unknown operation')
            except Exception as exc:connection.send(dict(error=repr(exc)))
    except Exception as exc:
        connection.send(dict(error=repr(exc)))
    finally:connection.close()


class PaidOracle:
    def __init__(self,label_file,journal_file,budget=105):
        ctx=mp.get_context('spawn');self.connection,child=ctx.Pipe()
        self.process=ctx.Process(target=serve,args=(child,str(label_file),str(journal_file),budget))
        self.process.start();child.close()
        self.paid=self._receive()['paid']

    def _receive(self):
        out=self.connection.recv()
        if 'error' in out:raise ValueError(out['error'])
        return out

    def query(self,index):
        self.connection.send(dict(op='query',index=index))
        r=self._receive()['record'];self.paid.append(r);return r['value']

    def metrics(self):
        self.connection.send(dict(op='metrics'));return self._receive()['metrics']

    def close(self):
        if self.process.is_alive():self.connection.send(dict(op='close'))
        self.process.join(timeout=10);self.connection.close()
        if self.process.is_alive():self.process.terminate();self.process.join()
