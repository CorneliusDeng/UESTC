import time

# 读取端口文件和流文件
def get_ports(ports_file):
    ports = []
    with open(ports_file) as f:
        next(f)
        for line in f:
            port_id, bandwidth = line.strip().split(',')
            ports.append({'id': int(port_id), 'bandwidth': int(bandwidth), 'queue': [], 'using': 0})
    return ports  

def read_flows(flows_file):
    flows = []
    with open(flows_file) as f:
        next(f)
        for line in f:
            flow_id, bandwidth, entry_time, transmission_time = line.strip().split(',')
            flows.append({'id': int(flow_id), 'bandwidth': int(bandwidth), 'entry_time': int(entry_time), 'transmission_time': int(transmission_time), 'port_id': -1})
    return flows

# 处理流并输出结果
def schedule_flows(ports, flows):
    result = []
    time = 0
    for flow in flows:
        min_port_id = None
        flow_id, bandwidth, arrive_time, duration = flow 
        # 找到第一个可用带宽满足flow的端口
        for port, port_bandwidth in ports:
            if port_bandwidth >= bandwidth:
                # 检查端口当前是否有空余带宽和是否有等待流
                avail_bandwidth = port_bandwidth  # 当前可用带宽
                queue = []  # 等待流队列
                for sent_flow in result:
                    if sent_flow[1] == port:
                        avail_bandwidth -= sent_flow[2]  # 减去已发送流带宽
                        if sent_flow[3] > arrive_time:  # 如果发送未完成,加入等待队列
                            queue.append(sent_flow)
                # 如果当前可用带宽满足并无等待流,开始发送
                if avail_bandwidth >= bandwidth and not queue:
                    send_time = max(arrive_time, result[-1][3] if result else 0)  # 开始发送时间
                    result.append((flow_id, port, bandwidth, send_time, send_time + duration))
                    break
                # 否则加入等待队列等待
                else:
                    queue.append(flow) 
        else:
            # 如果所有端口带宽都不满足,流无法发送
            pass
    return result

# 写入输出文件 
def write_result(result_file, result):
    with open('result.txt', 'w') as f:
        f.write('流id,端口id,开始发送时间\n')
        for flow in result:
            f.write(','.join(str(x) for x in flow[:3]) + '\n')
    

def main():
    port_file = '../data/port.txt'  # 端口文件
    flow_file = '../data/flow.txt'   # 流文件
    ports = read_ports(port_file)
    flows = read_flows(flow_file)
    result = schedule_flows(ports, flows)
    write_result('../data/result.txt', result)

if __name__ == "__main__":
    main()