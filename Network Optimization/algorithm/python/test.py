port_file = '../data/port.txt'  # 端口文件
flow_file = '../data/flow.txt'   # 流文件

# 读取端口信息
with open(port_file) as f:
    ports = []
    next(f)  # 跳过第一行
    for line in f:
        port_id, bandwidth = line.strip().split(',')
        ports.append({
            'id': int(port_id),
            'bandwidth': int(bandwidth),
            'current_bandwidth': 0,  # 当前端口正在使用的带宽
            'queue': []              # 端口的等待队列
        })

# 读取流信息
with open(flow_file) as f:
    flows = []
    next(f)
    for line in f:
        flow_id, bandwidth, arrive_time, duration = line.strip().split(',')
        flows.append({
            'id': int(flow_id),
            'bandwidth': int(bandwidth),
            'arrive_time': int(arrive_time),
            'duration': int(duration),
            'start_time': -1,   # 流的开始发送时间
            'port_id': -1      # 流分配的端口id
        })

flows.sort(key=lambda x: x['arrive_time'])  # 按到达时间排序
result = []  # 结果
time = 0     # 当前时间

for flow in flows:
    # 找到第一个可用带宽满足流需要的端口
    for port in ports:
        if (port['bandwidth'] - port['current_bandwidth']) >= flow['bandwidth']:
            flow['port_id'] = port['id']
            break
    min_port = None
    # 如果找到可用端口,开始发送流,否则放入端口等待队列
    if flow['port_id'] != -1:
        flow['start_time'] = time
        port['current_bandwidth'] += flow['bandwidth']
        result.append([flow['id'], flow['port_id'], flow['start_time']])
    else:
        # 从等待时间最短的端口队列中选一个放入
        min_port = min(ports, key=lambda x: x['queue'][0]['arrive_time'] if x['queue'] else float('inf'))
        min_port['queue'].append(flow)
    
    # 时间前进到下一条流的到达时间或者下一个流开始发送时间
    if min_port and min_port['queue']:
        time = min(flow['arrive_time'] + flow['duration'], min_port['queue'][0]['arrive_time'])
    else:
        time = flow['arrive_time'] + flow['duration']
    
    # 检查各个端口,若有流发送完毕,更新当前带宽和队列
    for port in ports:
        if port['queue'] and port['queue'][0]['start_time'] + port['queue'][0]['duration'] <= time:
            current_flow = port['queue'].pop(0)
            port['current_bandwidth'] -= current_flow['bandwidth']
            result.append([current_flow['id'], current_flow['port_id'], current_flow['start_time']])

# 输出结果到文件        
with open('../data/result.txt', 'w') as f:
    f.write('流id,端口id,开始发送时间\n')
    for flow_id, port_id, start_time in result:
        f.write(f'{flow_id},{port_id},{start_time}\n')