# dataset_tools
数据集工具合集

```bash
python main.py --dataset_type paddle_detection  --input_dataset_dir F:\数据集\VOC数据集\箱门检测数据集\ContainVOC  --save_dataset_dir E:\Data\VOC数据集\箱门检测数据集\ContainVOC --remove_classes "UPPEREND,DOOREND"
python main.py --dataset_type paddle_text_detection --input_dataset_dir F:\数据集\关键点检测数据集\箱号关键点数据集 --save_dataset_dir E:\Data\关键点检测数据集\箱号关键点数据集
python main.py --dataset_type paddle_text_detection --input_dataset_dir F:\数据集\关键点检测数据集\定制版箱号关键点数据集 --save_dataset_dir E:\Data\关键点检测数据集\定制版箱号关键点数据集
python main.py --dataset_type paddle_text_recognize --dataset_name "箱号数据集" --input_dataset_dir E:\Data\关键点检测数据集\箱号关键点数据集 --save_dataset_dir E:\Data\文本识别数据集\箱号文本识别数据集
```