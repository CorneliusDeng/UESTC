import json
import struct
import socket
import threading
import time


# 功能1：sum函数
def sum(conn, params, args):
    try:
        # 以逗号分隔获取两个加数
        a, b = str(params).split(',')
        if args == 'float':
            # 保留6位小数
            res = round(float(a) + float(b), 6)
        elif args == 'int':
            res = int(a) + int(b)
        send_result(conn, 'sum_result', res)
    except:
        send_result(conn, 'error', 'error')


# 功能2：uppsercase函数
def uppercase(conn, params, args):
    # print(str.upper(params))
    res = str.upper(params)
    send_result(conn, 'upper_result', res)


def handle_conn(conn, addr, handlers):
    print('client: {0} connect'.format(addr))
    # 循环读写
    while True:
        length_prefix = conn.recv(4)
        if not length_prefix:  # 连接关闭
            print('client: {0} close'.format(addr))
            conn.close()
            break
        length, = struct.unpack('I', length_prefix)
        body = conn.recv(length)
        request = json.loads(body)
        print('recv :', request)
        in_ = request['in']
        params = request['params']
        args = request['args']
        # 查找请求对应的处理函数
        handler = handlers[in_]
        # 用于测试并发、将其慢处理，使请求交叉通过
        # time.sleep(1.5)
        # 处理请求
        handler(conn, params, args)


# 执行handlers函数时为了保证多用户并发使用应该使用多线程
def loop(sock, handlers):
    while True:
        conn, addr = sock.accept()  # 接收连接
        t = threading.Thread(target=handle_conn, args=(conn, addr, handlers))
        t.start()


# 向sender发送响应体
def send_result(conn, out, result):
    response = json.dumps({'out': out, 'result': result})  # 响应消息体
    lenght_prefix = struct.pack('I', len(response))
    conn.sendall(lenght_prefix)
    conn.sendall(str.encode(response))


if __name__ == '__main__':
    # 创建一个 TCP 套接字
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # 打开 reuse addr 选项
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("localhost", 8080))  # 绑定端口
    sock.listen(1)  # 监听客户端连接

    print('listen')
    # 注册请求处理器。反射作用。根据请求执行响应函数体
    handlers = {
        'sum': sum,
        'uppercase': uppercase
    }
    loop(sock, handlers)
