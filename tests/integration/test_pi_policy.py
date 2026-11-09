import json
import os
from pathlib import Path
from types import SimpleNamespace
import pytest
from agent import pi


def resources(available=6,cpu=10,pressure=0,temp=45):
    return pi.Resources(int(available*pi.GIB),cpu,pressure,temp,0,4)


@pytest.mark.parametrize('size,gate,peak',[(500_000_000,2.5,1.5),(1_500_000_000,3.5,2.5),(3_000_000_000,5,3.5)])
def test_tiers(size,gate,peak):
    assert pi.admission(resources(max(gate,peak+1.5)),size,peak,False,0,2048)['state']=='allowed'
    assert pi.admission(resources(gate-.1),size,peak,False,0,2048)['state']=='queued'


@pytest.mark.parametrize('size',[4_000_000_000,7_000_000_000,30_000_000_000])
def test_large(size):
    with pytest.raises(pi.Blocked):pi.admission(resources(32),size,15,False,0,2048)


@pytest.mark.parametrize('r',[resources(cpu=71),resources(pressure=11),resources(temp=75),resources(available=1)])
def test_pressure(r):
    assert pi.admission(r,1_500_000_000,2.5,False,0,2048)['state']=='queued'


def test_residency():
    result=pi.admission(resources(2),1_500_000_000,2.5,True,int(2*pi.GIB),2048)
    assert result['state']=='allowed' and result['additional_gib']==.5
    assert pi.admission(resources(1.4),1_500_000_000,2.5,True,int(3*pi.GIB),2048)['state']=='queued'


def test_nested_cgroup(tmp_path):
    proc=tmp_path/'proc';cg=tmp_path/'cg';(proc/'self').mkdir(parents=True);(cg/'service').mkdir(parents=True)
    (proc/'meminfo').write_text('MemAvailable: 8000000 kB\nSwapFree: 99000000 kB\n')
    (proc/'self/cgroup').write_text('0::/service\n')
    (cg/'memory.max').write_text(str(4*pi.GIB));(cg/'memory.current').write_text(str(3*pi.GIB))
    (cg/'service/memory.max').write_text('max')
    assert pi.effective_available(proc,cg)==pi.GIB


def test_lock(tmp_path):
    lock=tmp_path/'lock';lock.touch()
    with pi.inference_slot(lock):
        with pytest.raises(pi.Blocked):
            with pi.inference_slot(lock,wait=0):pass
    with pi.inference_slot(lock,wait=0):pass


@pytest.mark.parametrize('url',['https://example.com','http://192.168.1.2:11434','http://127.0.0.1:11434@evil.test'])
def test_remote(url):
    with pytest.raises(pi.Blocked):pi.LocalOllama(url)


def fixture(monkeypatch,tmp_path):
    digest='a'*64;lock=tmp_path/'queue.lock';lock.touch();monkeypatch.setenv('RISARMS_INFERENCE_LOCK',str(lock))
    path=tmp_path/'models.json';path.write_text(json.dumps({'fixture:1.5b-q4':{'digest':digest,'quantization':'Q4_K_M','peak_gib':2.5,'contexts':[2048],'generation_verified':True}}));monkeypatch.setenv('ASCS_PI_MANIFEST',str(path))
    monkeypatch.setattr(pi,'sample_resources',lambda:resources())
    monkeypatch.setattr(pi,'deployment_supported',lambda:True)
    calls=[]
    class Client:
        def __init__(self,url):pass
        def request(self,path,body=None,**kw):
            calls.append((path,body))
            if path=='/api/tags':return {'models':[{'name':'fixture:1.5b-q4','digest':digest}]}
            if path=='/api/show':return {'template':'chat','capabilities':['completion'],'details':{'quantization_level':'Q4_K_M'},'model_info':{'general.parameter_count':1_500_000_000}}
            if path=='/api/ps':return {'models':[]}
            return {'done':True,'message':{'content':'print(1)'}}
    monkeypatch.setattr(pi,'LocalOllama',Client)
    args=SimpleNamespace(ui=False,tui=False,tasks=False,model='fixture:1.5b-q4',pi_status=False,task='make code',workspace=str(tmp_path),num_ctx=None,base_url=None)
    return args,calls,path


def test_generate(monkeypatch,tmp_path,capsys):
    args,calls,_=fixture(monkeypatch,tmp_path)
    assert pi.run(args)==0
    assert calls[-1][1]['options']=={'num_ctx':2048,'num_thread':2,'num_predict':1024}
    assert calls[-1][1]['keep_alive']==0 and 'tools' not in calls[-1][1]
    assert 'no project files changed' in capsys.readouterr().out
    assert not (tmp_path/'.ascs').exists()


@pytest.mark.parametrize('change',['digest','unverified','context'])
def test_unapproved(monkeypatch,tmp_path,change):
    args,calls,path=fixture(monkeypatch,tmp_path);data=json.loads(path.read_text())
    if change=='digest':data[args.model]['digest']='b'*64
    elif change=='unverified':data[args.model]['generation_verified']=False
    else:args.num_ctx=65536
    path.write_text(json.dumps(data));assert pi.run(args)==1
    assert not any(p=='/api/chat' for p,_ in calls)


def test_pressure_no_generate(monkeypatch,tmp_path):
    args,calls,_=fixture(monkeypatch,tmp_path);monkeypatch.setattr(pi,'sample_resources',lambda:resources(cpu=90))
    assert pi.run(args)==75
    assert not any(p=='/api/chat' for p,_ in calls)


def test_cancel(monkeypatch,tmp_path):
    args,_,_=fixture(monkeypatch,tmp_path)
    monkeypatch.setattr(pi,'sample_resources',lambda:(_ for _ in ()).throw(KeyboardInterrupt()))
    assert pi.run(args)==130
    with pi.inference_slot(Path(os.environ['RISARMS_INFERENCE_LOCK']),wait=0):pass


def test_guard(monkeypatch,tmp_path):
    monkeypatch.setenv('ASCS_PI_MODE','1')
    from agent.config import AgentConfig
    from agent.core.loop import AgentLoop
    with pytest.raises(RuntimeError,match='tools disabled'):AgentLoop(AgentConfig(workspace=tmp_path),None,None)
    from agent.main import main
    assert main(['--workspace',str(tmp_path),'--mode','auto','--','edit project'])==1


def test_explicit_switch_only_approved_resident(monkeypatch,tmp_path):
    args,calls,path=fixture(monkeypatch,tmp_path)
    data=json.loads(path.read_text());data['old:tag']={'digest':'b'*64};path.write_text(json.dumps(data))
    args.pi_unload_previous=True
    original=pi.LocalOllama
    class Switching(original):
        unloaded=False
        def request(self,path,body=None,**kw):
            if path=='/api/ps':
                return {'models':[] if self.unloaded else [{'name':'old:tag','digest':'b'*64}]}
            if path=='/api/generate':
                self.unloaded=True;calls.append((path,body));return {'done':True}
            return super().request(path,body,**kw)
    monkeypatch.setattr(pi,'LocalOllama',Switching)
    assert pi.run(args)==0
    assert ('/api/generate',{'model':'old:tag','keep_alive':0}) in calls


def test_unknown_resident_not_unloaded(monkeypatch,tmp_path):
    args,calls,_=fixture(monkeypatch,tmp_path);args.pi_unload_previous=True
    original=pi.LocalOllama
    class Other(original):
        def request(self,path,body=None,**kw):
            if path=='/api/ps':return {'models':[{'name':'unrelated','digest':'b'*64}]}
            return super().request(path,body,**kw)
    monkeypatch.setattr(pi,'LocalOllama',Other)
    assert pi.run(args)==1
    assert not any(path in ('/api/generate','/api/chat') for path,_ in calls)


def test_cloud_tag_never_receives_prompt(monkeypatch,tmp_path):
    args,calls,_=fixture(monkeypatch,tmp_path)
    original=pi.LocalOllama
    class Cloud(original):
        def request(self,path,body=None,**kw):
            result=super().request(path,body,**kw)
            if path=='/api/tags':result['models'][0]['remote_host']='https://cloud.example'
            return result
    monkeypatch.setattr(pi,'LocalOllama',Cloud)
    assert pi.run(args)==1
    assert not any(path=='/api/chat' for path,_ in calls)


def test_busy_queue_reports_transient_exit(monkeypatch,tmp_path,capsys):
    args,_,_=fixture(monkeypatch,tmp_path)
    from contextlib import contextmanager
    @contextmanager
    def busy(*a,**kw):
        raise pi.QueueBusy('busy')
        yield
    monkeypatch.setattr(pi,'inference_slot',busy)
    assert pi.run(args)==75
    assert '"state": "queued"' in capsys.readouterr().out
