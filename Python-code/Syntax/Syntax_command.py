# Start matlab
eng = matlab.engine.start_matlab()
eng.eval("addpath('.\Matlab_lib');")
eng.eval("addpath('.\Matlab_lib\mPWM');")
 ret = eng.dec2base(2**60,16,stdout=out,stderr=err)
print(err.getvalue())
print(out.getvalue())