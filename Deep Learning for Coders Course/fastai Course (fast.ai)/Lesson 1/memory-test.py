"""testing vram in pytorch cuda
every time a variable is put inside a container in python, to remove it completely
one needs to delete variable and container,
this can be problematic when using pytorch cuda if one doesnt clear all containers
Three tests:
>>> python memory_tests list
    # creates 2 tensors puts them in a list, modifies them in place, deletes them
    # in place mod changes original tensors
    # list and both tensors need to be deleted
>>> python memory_tests scoped list
    # creates 2 tensors puts them in a list, passes to function that modifies them in place, deletes them
    # in place mod changes original tensors
    # list, list returned by function and both tensors need to be deleted
>>> python memory_tests dict
    # creates 2 tensors puts them in a dict, modifies them in place, deletes them
    # in place mod changes original tensors
    # list and both tensors need to be deleted
Or, the simplest solution is to encapsulate all operations in a module, then delete the module output, dictionary,
tuple, class or dict. If there is no leaking within the module then everything will be properly cleaned.
Unlike torch types however, clobbering an input list with an output list wont delete the underlying data and will
render it inaccessible.
Example
>>> t0 = torch.rndn((1,3,1024,1024), device="cuda")
>>> t1 = torch.rndn((1,3,1024,1024), device="cuda")
>>> data = (t0, t1)
>>> data = MyModule.myaction(data)
    # if operations are not inplace,
    # this will lead to a leak until your current module goes out of scope
>>> data1 = MyModule.myaction(data)
>>> del data
>>> del data1
    # this should work
"""
import sys
import time
import subprocess as sp
import torch
import math

class color:
    B="\033[1m"
    K="\u001b[30m"
    R="\u001b[31m"
    G="\u001b[32m"
    Y="\u001b[33m"
    W="\033[0m"

def report(msg, nv,pt,tm):
    nv.append(int(sp.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'], encoding='utf-8').split('\n')[0]))
    tm.append(torch.cuda.max_memory_allocated()//2**20)
    pt.append(torch.cuda.memory_allocated()//2**20)
    print(f"{color.Y}{msg}{color.W}")
    print(f" nvida smi:\t{nv[-1]} MB\ttotal torch: {nv[-1] - nv[0]}")
    # print(f" torch mem :\t{pt[-1]} MB\ttotal torch: {pt[-1] - pt[0]}")
    # print(f" torch maxmem:\t{tm[-1]} MB\ttotal torch: {tm[-1] - tm[0]}")
    return nv, tm, pt


def init_cuda(msg=""):
    torch.cuda.init()
    torch.empty(1, device="cuda")
    return report("cudainit1 %s%s%s"%(color.B, msg, color.W), [], [], [])

def tensorsize(tensor):
    return math.ceil(len(tensor.view(-1))*tensor.element_size()/2**20)

def test_dictionary_global():
    nv,pt,tm = init_cuda("TESTING DICT global")

    # build tensors
    t0 = torch.randn((1, 3, 1014, 1014), device="cuda")
    nv,pt,tm = report("added one vector of size %d MB"%(tensorsize(t0)), nv, pt, tm)
    t1 = torch.randn((1, 3, 214, 2014), device="cuda")
    nv,pt,tm = report("added second vector of size %d MB"%(tensorsize(t1)), nv, pt, tm)

    # add tensors to dict
    f = {"t0":t0, "t1":t1}
    nv,pt,tm = report("added vectors to dict", nv, pt, tm)

    f["t0"].add_(1).mul_(0.001)
    _b = bool((t0 == f["t0"]).all().item())
    c = [color.R, color.G][_b]
    print(c, "changed by reference:", _b, color.W)

    del t0
    del t1
    torch.cuda.empty_cache()
    nv,pt,tm = report("deleting original sensors", nv,pt,tm)
    _cleared = bool((nv[-1]-nv[0]) < 1)
    c=[color.R, color.G][_cleared]
    print(c, "Memory was cleared :", _cleared, color.W)

    del f
    torch.cuda.empty_cache()
    nv,pt,tm = report("deleting dictionary", nv,pt,tm)
    _cleared = bool((nv[-1]-nv[0]) < 1)
    c=[color.R, color.G][_cleared]
    print(c, "Memory was cleared :", _cleared, color.W)
    print("Afd")
    
def test_list_global():
    nv,pt,tm = init_cuda("TESTING LIST global")

    # build tensors
    t0 = torch.randn((1, 3, 1014, 1014), device="cuda")
    nv,pt,tm = report("added one vector of size %d MB"%(tensorsize(t0)), nv, pt, tm)
    t1 = torch.randn((1,3, 214, 2014), device="cuda")
    nv,pt,tm = report("added second vector of size %d MB"%(tensorsize(t1)), nv, pt, tm)

    # add tensors to dict
    f = [t0, t1]
    nv,pt,tm = report("added vectors to list", nv, pt, tm)

    f[0].add_(1).mul_(0.001)
    _b = bool((t0 == f[0]).all().item())
    c = [color.R, color.G][_b]
    print(c, "changed by reference:", _b, color.W)

    del t0
    del t1
    torch.cuda.empty_cache()
    nv,pt,tm = report("deleting original sensors", nv, pt, tm)
    _b = bool((nv[-1]-nv[0]) < 1)
    c = [color.R, color.G][_b]
    print(c, "Memory was cleared :", _b, color.W)

    del f
    torch.cuda.empty_cache()
    nv,pt,tm = report("deleting list", nv, pt, tm)
    _b = bool((nv[-1]-nv[0]) < 1)
    c = [color.R, color.G][_b]
    print(c, "Memory was cleared :", _b, color.W)


def scope_list(f):
    f[0].add_(1).mul_(0.001)
    return f
   
def test_list_scoped():
    nv,pt,tm = init_cuda("TESTING LIST SCOPED")

    # build tensors
    t0 = torch.randn((1, 3, 1014, 1014), device="cuda")
    nv,pt,tm = report("added one vector of size %d MB"%(tensorsize(t0)), nv, pt, tm)
    t1 = torch.randn((1,3, 214, 2014), device="cuda")
    nv,pt,tm = report("added second vector of size %d MB"%(tensorsize(t1)), nv, pt, tm)

    # add tensors to dict
    f = [t0, t1]
    nv,pt,tm = report("added vectors to list", nv, pt, tm)

    f1 = scope_list(f)
    _b = bool((t0 == f[0]).all().item() and (t0 == f1[0]).all().item())
    c = [color.R, color.G][_b]
    print(c, "changed by reference:", _b, color.W)

    del t0
    del t1
    torch.cuda.empty_cache()
    nv,pt,tm = report("deleting original sensors", nv, pt, tm)
    _b = bool((nv[-1]-nv[0]) < 1)
    c = [color.R, color.G][_b]
    print(c, "Memory was cleared :", _b, color.W)

    del f
    torch.cuda.empty_cache()
    nv,pt,tm = report("deleting original list", nv, pt, tm)
    _b = bool((nv[-1]-nv[0]) < 1)
    c = [color.R, color.G][_b]
    print(c, "Memory was cleared :", _b, color.W)

    del f1
    torch.cuda.empty_cache()
    nv,pt,tm = report("deleting returned list", nv, pt, tm)
    _b = bool((nv[-1]-nv[0]) < 1)
    c = [color.R, color.G][_b]
    print(c, "Memory was cleared :", _b, color.W)

if __name__ == "__main__":

    largs = len(sys.argv) > 1

    if largs and sys.argv[1].lower()[0] == "l":
        test_list_global()
    elif largs and sys.argv[1].lower()[0] == "s":
        test_list_scoped()
    else:
        test_dictionary_global()