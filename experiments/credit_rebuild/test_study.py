import numpy as np
from study import bp, forward, local, signals, init

def fixture():
    rng=np.random.default_rng(8)
    w=[rng.normal(size=(3,4))*.2,rng.normal(size=(4,3))*.2,rng.normal(size=(3,2))*.2]
    return w,rng.normal(size=(5,3)),rng.normal(size=(5,2))

def test_bp_finite_difference():
    w,x,y=fixture(); g=bp(w,x,y); eps=1e-6
    for i in range(3):
        for index in np.ndindex(w[i].shape):
            old=w[i][index]
            w[i][index]=old+eps; plus=.5*np.sum((forward(w,x)[1]-y)**2)/len(x)
            w[i][index]=old-eps; minus=.5*np.sum((forward(w,x)[1]-y)**2)/len(x)
            w[i][index]=old
            np.testing.assert_allclose(g[i][index],(plus-minus)/(2*eps),atol=1e-8)

def test_first_step_equivalence():
    w,x,y=fixture(); s=[np.ones(4),np.ones(3)]
    a=local(w,x,y,s,alpha=0,budget=1)[0]
    b=local(w,x,y,s,alpha=.5,budget=1)[0]
    for u,v in zip(a,b): np.testing.assert_array_equal(u,v)

def test_scaled_activity_gradient_finite_difference():
    w,x,y=fixture(); h,_=forward(w,x)
    h=[h[0],h[1]+.1,h[2]-.2]
    s=[np.array([.75,1.,1.2,1.5]),np.array([1.,.8,1.3])]
    dual=[np.ones_like(h[1])*.1,np.ones_like(h[2])*.2]
    p,r,q=signals(w,h,dual,s)
    analytical=[q[0]-(q[1]*(1-p[1]**2))@w[1].T,q[1]+(h[-1]@w[-1]-y)@w[-1].T]
    def energy():
        _,r,_=signals(w,h,dual,s)
        return .5*np.sum((h[-1]@w[-1]-y)**2)+sum(np.sum(d*e/z+.5*(e/z)**2) for d,e,z in zip(dual,r,s))
    for i in [1,2]:
        for index in np.ndindex(h[i].shape):
            old=h[i][index]; eps=1e-6
            h[i][index]=old+eps; plus=energy()
            h[i][index]=old-eps; minus=energy(); h[i][index]=old
            np.testing.assert_allclose(analytical[i-1][index],(plus-minus)/(2*eps),atol=1e-8)

def test_mask_and_known_sign_initialization():
    m=[np.array([[True,False],[True,True]])]*3
    s=[np.array([[1,1],[-1,0]])]*3
    for w,mask,sign in zip(init(m,s,1),m,s):
        assert np.all(w[~mask]==0)
        assert np.all((w*sign)[sign!=0]>=0)
