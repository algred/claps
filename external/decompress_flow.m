function flow = decompress_flow(flow_int, flow_frac)
flow.u = double(flow_int.u) - 127 + double(flow_frac.u) / 100;
flow.v = double(flow_int.v) - 127 + double(flow_frac.v) / 100;
flow.hu = double(flow_int.hu) - 127 + double(flow_frac.hu) / 100;
flow.hv = double(flow_int.hv) - 127 + double(flow_frac.hv) / 100;
flow.bu = double(flow_int.bu) - 127 + double(flow_frac.bu) / 100;
flow.bv = double(flow_int.bv) - 127 + double(flow_frac.bv) / 100;
end
