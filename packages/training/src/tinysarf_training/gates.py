"""Pure fail-closed release gates; evidence collection is in promote.py."""
import math

def assess(candidate,active,policy):
    reasons=[]
    def require(ok,reason):
        if not ok: reasons.append(reason)
    def measured(x): return isinstance(x,(float,int)) and not isinstance(x,bool) and math.isfinite(x)
    require(candidate.get('integrityVerified') is True,'Model and dataset integrity must be verified by the promotion command')
    require(candidate.get('cleanReproduction') is True,'Clean-clone reproduction is missing')
    require(candidate.get('humanReviewedFixtures') is True,'50 contract fixtures require independent human review')
    require(candidate.get('independentGold') is True,'Independent human gold evaluation is missing')
    require(candidate.get('goldBeatsBaselines') is True,'Gold accuracy must beat both fixed baselines')
    require(candidate.get('browserContractPassed') is True,'Real-browser input, device-loss, and maximum-shape contracts must pass')
    require(candidate.get('git',{}).get('commit') is not None and candidate.get('git',{}).get('dirty') is False,'Clean Git provenance is required')
    require(bool(candidate.get('verificationDigest')),'Frozen verification digest is required')
    metrics=candidate.get('metrics',{})
    for key,floor in policy['floors'].items():
        value=metrics.get(key); require(measured(value) and value>=floor,f'{key} must meet floor {floor}')
    for name in ['gold-natural','gold-balanced']:
        for key,floor in policy['floors'].items():
            value=candidate.get('goldMetrics',{}).get(name,{}).get(key)
            require(measured(value) and value>=floor,f'{name}/{key} must meet floor {floor}')
    for slice_ in policy['protectedSlices']:
        require(candidate.get('slices',{}).get(slice_,{}).get('count',0)>0,f'Protected slice {slice_} has no evaluation evidence')
    for slice_ in policy.get('protectedInputSlices',[]):
        check=candidate.get('inputSlices',{}).get(slice_,{})
        require(check.get('count',0)>0 and check.get('accuracy')==1,f'{slice_} input rejection must be fully verified')
    require(candidate.get('warmupStable') is True,'Warmup stability was not established')
    require(candidate.get('fullBrowserProtocol') is True,'Complete reference-browser measurement protocol is required')
    if active is not None:
        require(candidate.get('verificationDigest')==active.get('verificationDigest'),'Candidate and active verification examples differ')
        require(candidate.get('normalizationVersion')==active.get('normalizationVersion'),'Normalization versions differ')
        weights=policy['compositeWeights']
        if all(measured(metrics.get(k)) and measured(active.get('metrics',{}).get(k)) for k in weights):
            require(sum(metrics[k]*w for k,w in weights.items())>sum(active['metrics'][k]*w for k,w in weights.items()),'Composite does not strictly improve')
        else: require(False,'Composite metrics are incomplete')
        for key in policy['floors']:
            c=metrics.get(key);a=active.get('metrics',{}).get(key)
            delta=policy['maximumTop1Regression'] if key=='fullAnalysisTop1' else policy['maximumSliceRegression']
            require(measured(c) and measured(a) and c>=a-delta,f'{key} regresses beyond allowed delta')
        for slice_ in policy['protectedSlices']:
            c=candidate.get('slices',{}).get(slice_,{});a=active.get('slices',{}).get(slice_,{})
            require(c.get('digest') is not None and c.get('digest')==a.get('digest'),f'{slice_} example identities differ')
            for key in policy['floors']:
                cv=c.get('metrics',{}).get(key);av=a.get('metrics',{}).get(key)
                require(measured(cv) and measured(av) and cv>=av-policy['maximumSliceRegression'],f'{slice_}/{key} regresses or is unmeasured')
        c=candidate.get('warmBatch128P95Ms');a=active.get('warmBatch128P95Ms')
        require(measured(c) and measured(a) and c<=a*(1+policy['maximumWarmLatencyRegression']),'Warm latency regression exceeds tolerance')
        require(candidate.get('referenceDevice') is not None and candidate.get('referenceDevice')==active.get('referenceDevice'),'Performance was measured on different devices')
    p=candidate.get('parity',{})
    require(measured(p.get('floatMaxAbsoluteError')) and p['floatMaxAbsoluteError']<=policy['floatMaxAbsoluteError'],'Float reference parity failed or unmeasured')
    require(measured(p.get('gpuArgmaxAgreement')) and p['gpuArgmaxAgreement']>=policy['minimumQuantizedGPUArgmaxAgreement'],'Quantized GPU argmax parity failed or unmeasured')
    require(p.get('deterministicRuns',0)>=policy['deterministicWarmRuns'] and p.get('deterministic') is True,'Warm determinism must pass 100 repeated runs')
    require(p.get('finite') is True,'All parity outputs must be finite')
    for key,maximum in policy['budgets'].items():
        value=candidate.get(key);require(measured(value) and value<=maximum,f'{key} exceeds budget or is unmeasured')
    require(candidate.get('coldRuns',0)>=policy['minimumColdRuns'],'At least 20 independent cold runs are required')
    require(candidate.get('warmups',0)>=policy['minimumWarmups'],'At least 20 warmups are required')
    require(candidate.get('warmIterations',0)>=policy['minimumWarmIterations'],'At least 100 warm measurements per shape are required')
    require(candidate.get('warmRepeats',0)>=policy['minimumWarmRepeats'],'At least three complete warm repetitions are required')
    require(candidate.get('measuredBatchAdvantage') is True,'No measured practical-batch WebGPU advantage')
    return {'eligible':not reasons,'reasons':reasons}
