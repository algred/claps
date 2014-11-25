function [flow_int, flow_frac] = compress_flow(flow)
flow_int.u = uint8(max(0, min(255, floor(flow.u) + 127))); 
flow_frac.u = uint8(max(0, min(255, round((flow.u - floor(flow.u)) * 100)))); 

flow_int.v = uint8(max(0, min(255, floor(flow.v) + 127))); 
flow_frac.v = uint8(max(0, min(255, round((flow.v - floor(flow.v)) * 100)))); 

flow_int.hu = uint8(max(0, min(255, floor(flow.hu) + 127))); 
flow_frac.hu = uint8(max(0, min(255, round((flow.hu - floor(flow.hu)) * 100)))); 

flow_int.hv = uint8(max(0, min(255, floor(flow.hv) + 127))); 
flow_frac.hv = uint8(max(0, min(255, round((flow.hv - floor(flow.hv)) * 100))));

flow_int.bu = uint8(max(0, min(255, floor(flow.bu) + 127))); 
flow_frac.bu = uint8(max(0, min(255, round((flow.bu - floor(flow.bu)) * 100)))); 

flow_int.bv = uint8(max(0, min(255, floor(flow.bv) + 127))); 
flow_frac.bv = uint8(max(0, min(255, round((flow.bv - floor(flow.bv)) * 100)))); 

end
