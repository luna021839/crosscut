from pathlib import Path
import hashlib
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from crosscut.onnx_cpu_inference import OnnxCpuNetwork, OnnxCpuBackendUnavailable


class FakeSession:
    def __init__(self, path, *, sess_options, providers):
        self.options=sess_options
        self.providers=providers
        self.outputs=['a','b']
        self.inputs=[SimpleNamespace(name='input',type='tensor(float)',shape=[1,2])]
        self.calls=[]
        self.invalid=False
    def get_providers(self): return self.providers
    def get_inputs(self): return self.inputs
    def get_outputs(self): return [SimpleNamespace(name=n) for n in self.outputs]
    def run(self,names,feed):
        self.calls.append((names,feed))
        if self.invalid: return [np.asarray([np.nan],np.float32) for _ in names]
        return [feed['input']+(1 if name=='a' else 2) for name in names]


class OnnxCpuInferenceTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path=Path(self.directory.name)/'model.onnx'
        self.path.write_bytes(b'bound fixture model')
        self.sha=hashlib.sha256(self.path.read_bytes()).hexdigest()
        self.sessions=[]
        def session(*args,**kwargs):
            value=FakeSession(*args,**kwargs); self.sessions.append(value); return value
        self.runtime=SimpleNamespace(__version__='fixture-1',SessionOptions=SimpleNamespace,
                                     get_available_providers=lambda:['CPUExecutionProvider'],InferenceSession=session)
        self.patch=patch('crosscut.onnx_cpu_inference.import_module',return_value=self.runtime)
        self.patch.start(); self.addCleanup(self.patch.stop)
    def build(self,**kwargs):
        return OnnxCpuNetwork(self.path,expected_sha256=self.sha,output_names=('b','a'),**kwargs)

    def test_explicit_cpu_and_output_order_preserve_caller_contract(self):
        net=self.build()
        net.setInput(np.asarray([[3.,4.]],np.float32))
        values=net.forward(('b','a'))
        np.testing.assert_array_equal(values[0],[[5,6]])
        np.testing.assert_array_equal(values[1],[[4,5]])
        self.assertEqual(net.getUnconnectedOutLayersNames(),('b','a'))
        self.assertEqual(self.sessions[0].providers,['CPUExecutionProvider'])
        self.assertEqual(self.sessions[0].options.intra_op_num_threads,2)
        self.assertEqual(self.sessions[0].options.inter_op_num_threads,1)

    def test_input_is_owned_and_failed_input_cannot_reuse_previous_tensor(self):
        net=self.build(); data=np.asarray([[1.,2.]],np.float32)
        net.setInput(data); data[:]=99
        np.testing.assert_array_equal(net.forward(('a',))[0],[[2,3]])
        for bad in (np.ones((1,3),np.float32),np.ones((1,2),np.float64),np.asarray([[np.nan,0]],np.float32)):
            with self.assertRaises(ValueError): net.setInput(bad)
            with self.assertRaises(ValueError): net.forward(('a',))

    def test_bad_sha_rejected_before_backend_import(self):
        with self.assertRaises(ValueError):
            OnnxCpuNetwork(self.path,expected_sha256='0'*64,output_names=('a','b'))
        self.assertFalse(self.sessions)

    def test_missing_dependency_or_cpu_is_not_silent_opencv_fallback(self):
        with patch('crosscut.onnx_cpu_inference.import_module',side_effect=ImportError):
            with self.assertRaises(OnnxCpuBackendUnavailable): self.build()
        self.runtime.get_available_providers=lambda:['CoreMLExecutionProvider']
        with self.assertRaises(OnnxCpuBackendUnavailable): self.build()

    def test_unknown_outputs_multi_default_and_nonfinite_are_rejected(self):
        net=self.build(); net.setInput(np.ones((1,2),np.float32))
        for names in (None,(),('missing',),('a','a'),'a'):
            with self.assertRaises(ValueError): net.forward(names)
        self.sessions[0].invalid=True
        with self.assertRaises(ValueError): net.forward(('a',))

    def test_runtime_model_metadata_and_provider_must_match(self):
        original=self.runtime.InferenceSession
        for mutation in (lambda s:setattr(s,'providers',['CoreMLExecutionProvider']),
                         lambda s:setattr(s,'inputs',[]),
                         lambda s:setattr(s,'outputs',['different']),
                         lambda s:setattr(s.inputs[0],'type','tensor(double)')):
            def create(*args,**kwargs):
                session=original(*args,**kwargs); mutation(session); return session
            self.runtime.InferenceSession=create
            with self.assertRaises(ValueError): self.build()

    def test_model_change_during_session_load_is_rejected(self):
        original=self.runtime.InferenceSession
        def create(*args,**kwargs):
            session=original(*args,**kwargs); self.path.write_bytes(b'changed model'); return session
        self.runtime.InferenceSession=create
        with self.assertRaises(ValueError): self.build()

    def test_single_default_output_and_fingerprint_are_explicit(self):
        original=self.runtime.InferenceSession
        def create(*args,**kwargs):
            session=original(*args,**kwargs); session.outputs=['a']; return session
        self.runtime.InferenceSession=create
        net=OnnxCpuNetwork(self.path,expected_sha256=self.sha,output_names=('a',))
        net.setInput(np.ones((1,2),np.float32))
        np.testing.assert_array_equal(net.forward(),[[2,2]])
        fingerprint=net.fingerprint()
        self.assertFalse(fingerprint['bitwise_opencv_cache_compatibility'])
        self.assertEqual(fingerprint['model_sha256'],self.sha)
        self.assertEqual(fingerprint['runtime_version'],'fixture-1')
        fingerprint['output_names'].clear()
        self.assertEqual(net.fingerprint()['output_names'],['a'])


if __name__=='__main__': unittest.main()
