AI infra 四类方向

1. 训练基础设施

​	数据：数据加载、预处理、shuffler

​	计算：分布式/单机 、混合精度

​	可靠性： Checkpoint、断点恢复

​	调度：训练任务排队



2. 推理基础设施

   Runtime：RNNX Runtime TensorRT

   计算： 算子库、图优化、Operator

   调度：

3. 资源基础设施

   部署、运维

4. AI编译器

   图优化、常量编译