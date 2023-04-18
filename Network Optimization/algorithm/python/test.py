import pulp

# 读取端口文件
def read_port_file(filename):
    ports = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            port_id, bandwidth = map(int, line.strip().split(','))
            ports[port_id] = bandwidth
    return ports

# 读取流文件
def read_flow_file(filename):
    flows = {}
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines[1:]:
            flow_id, bandwidth, enter_time, send_time = map(int, line.strip().split(','))
            flows[flow_id] = {
                'bandwidth': bandwidth,
                'enter_time': enter_time,
                'send_time': send_time
            }
    return flows

# 线性规划求解
def linear_programming(ports, flows):
    # 创建线性规划模型
    model = pulp.LpProblem('Flow Scheduling', pulp.LpMinimize)

    # 创建决策变量
    ports_vars = {}
    flows_vars = {}
    for port_id in ports:
        ports_vars[port_id] = pulp.LpVariable(f'port_{port_id}', lowBound=0, cat='Integer')
    for flow_id in flows:
        flows_vars[flow_id] = {}
        for port_id in ports:
            flows_vars[flow_id][port_id] = pulp.LpVariable(f'flow_{flow_id}_port_{port_id}', lowBound=0, cat='Binary')

    # 添加约束条件
    for port_id in ports:
        model += sum(flows_vars[flow_id][port_id] * flows[flow_id]['bandwidth'] for flow_id in flows) <= ports[port_id] * ports_vars[port_id]
    for flow_id in flows:
        model += sum(flows_vars[flow_id][port_id] for port_id in ports) == 1
        for port_id in ports:
            model += flows_vars[flow_id][port_id] * flows[flow_id]['send_time'] + ports_vars[port_id] >= flows[flow_id]['enter_time']

    # 设置目标函数
    model += sum(ports_vars.values())

    # 求解线性规划
    model.solve()

    # 解析结果
    result = []
    for flow_id in flows:
        for port_id in ports:
            if flows_vars[flow_id][port_id].value() == 1:
                result.append((flow_id, port_id, int(pulp.value(ports_vars[port_id]) - flows[flow_id]['send_time'])))
                break

    result = sorted(result, key=lambda x: x[0])
    return result

# 主函数
def main():
    # 读取端口文件和流文件
    ports = read_port_file('../data/port.txt')
    flows = read_flow_file('../data/flow.txt')

    # 线性规划求解
    result = linear_programming(ports, flows)

    # 输出结果
    with open('../data/result.txt', 'w') as f:
        f.write('流id,端口id,开始发送时间\n')
        for r in result:
            f.write(f'{r[0]},{r[1]},{r[2]}\n')

if __name__ == '__main__':
    main()