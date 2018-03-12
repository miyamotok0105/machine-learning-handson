
#クイックスタート

    python -m pip install --upgrade pip
    python -m pip install grpcio
    python -m pip install grpcio-tools

    git clone -b v1.10.x https://github.com/grpc/grpc
    cd grpc/examples/python/helloworld

    python greeter_server.py
    python greeter_client.py


serverを起動した状態でclientを動かすと    
[Greeter client received: Hello, you!]が返ってくる。    

