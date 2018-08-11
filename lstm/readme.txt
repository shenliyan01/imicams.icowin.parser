/v1 代码
	main.py
	dataloader.py
	model.py
	train_test.py
	test_n.txt	测试结果，形式word1|tag1|tag2\nword2, tag1是预测输出，tag2是gold
	fold_res.json	结果统计，f1, p, r
/data 数据
	crf_1001_2000.txt	原始文件
	crf_1001_2000.txt	处理为每个句子一行
	word2idx.json	单词和idx的映射词典
	label2idx.json	label和idx的映射词典

训练参数命令
python3 main.py --help
主要参数说明：
	--n_fold	n折交叉验证，1，2，3, ...
	--prop		n_fold为1的时候有效，训练集和测试集的比例

数据说明。
	data最后部分的数据比较特殊，当最后一部分作为测试集的时候，性能较差。