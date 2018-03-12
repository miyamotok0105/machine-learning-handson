
#grpcの基礎

https://grpc.io/docs/tutorials/basic/python.html#defining-the-service

- .protoファイルを定義
- 生成サーバーとクライアントコードがプロトコルバッファコンパイラーを使用。
- pythonでgRPC APIを使うシンプルなクライアントとサーバーを書く。

    python -m pip install --upgrade pip
    python -m pip install grpcio
    python -m pip install grpcio-tools

    git clone -b v1.10.x https://github.com/grpc/grpc
    cd grpc/examples/python/route_guide

  

# サービスを定義

サービスの特有の名前を.protoファイルにつける

    service RouteGuide {
       // (Method definitions not shown)
    }




