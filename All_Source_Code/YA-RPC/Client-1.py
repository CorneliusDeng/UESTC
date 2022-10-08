import json, time, struct, socket


def rpc(sock, in_, params, args=None):
    # 将请求序列化打包
    request = json.dumps({"in": in_, "params": params, 'args': args})
    # 把数据封装成字符串
    length_prefix = struct.pack('I', len(request))
    # 发送请求和请求体
    sock.sendall(length_prefix)
    sock.sendall(str.encode(request))
    print('send data:', str.encode(request))

    # 等待接受响应
    try:
        # 设置响应时间以实现At-least-once语义
        sock.settimeout(60)
        # 接受响应并且得到响体
        length_prefix = sock.recv(4)
        length, = struct.unpack('I', length_prefix)
        # 响应消息体
        body = sock.recv(length)
        print('receive:', body)
        response = json.loads(body)
        # 返回响应类型和结果
        return response['out'], response['result']
    # 响应时间超过时、重新发送请求
    except Exception as e:
        print(e)
        return rpc(sock, in_, params, args)


if __name__ == '__main__':
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(('localhost', 8080))

    # 测试并发请求
    print('并发测试sum')
    out, result = rpc(s, 'sum', '1,2', 'int')
    out, result = rpc(s, 'sum', '6.666,7.777', 'float')

    s.close()