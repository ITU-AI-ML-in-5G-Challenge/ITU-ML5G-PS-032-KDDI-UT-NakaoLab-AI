# itu-ml-challenge
./conf/local_conf.yaml 里有json文件的路径，这个配置文件可以随意添加和修改属性

# label type summary
- 0: ixnetwork-traffic-start
- 1: node-down
- 2: node-up
- 3: interface-down
- 4: interface-up
- 5: tap-loss-start
- 6: tap-loss-stop
- 7: tap-delay-start
- 8: tap-delay-stop
- 9: ixnetwork-bgp-injection-start
- 10: ixnetwork-bgp-injection-stop
- 11: ixnetwork-bgp-hijacking-start
- 12: ixnetwork-bgp-hijacking-stop
- 13: ixnetwork-traffic-stop

# CSV生成方法
在 json_extract_v2.py 里面的main方法里选择一个你要生成的csv文件（其他两个注释掉即可），然后在local_conf.yaml里指定你的数据源路径，运行json_extract_v2.py可以生成对应的csv，倒数第三个字段是时间戳，最后两个字段是label

# 字段分析图的使用方法，打开 xxx_feature_analysis.ipynb， 配置指定的csv路径（上面刚刚生成的csv），运行即可
