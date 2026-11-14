# 青年创业扶持门槛的模糊断点与识别诊断

本项目提供资格及前定评分关联相关资料的业务摘要，并在保留既有摘要的基础上提供模糊断点（FRD）局部效应分析。

app/catalog.py 按集合和记录键解析修订，拒绝同修订冲突；app/baseline.py 承载当前业务计算；app/service.py 输出摘要及输入摘要哈希。
app/frd.py 为独立的模糊断点入口，不替换既有摘要：联合估计结果跳跃与领取跳跃的三明治协方差，弱识别时给 Anderson–Rubin 接受集合而非窄正态区间，并附密度/堆积/离散支撑诊断与带宽、甜甜圈、窗口敏感性。
数据字段及接口见 [资料契约](docs/data-contract.md)。fixtures/sample.json 是合成小数据，fixtures/demo_frd.json 是可完整运行断点分析的合成数据，均不对应真实村庄或个人。
运行环境为 Python 3.14，所有模块仅使用标准库。

读取并计算当前摘要（保持不变）：

    python -m app.service fixtures/sample.json

模糊断点分析与识别诊断：

    python -m app.frd fixtures/demo_frd.json
    python -m app.frd fixtures/demo_frd.json --degree 2 --kernel uniform --boot-reps 399
    python -m app.frd fixtures/sample.json --no-sensitivity   # 小样本下应返回 not_identified

数值管线的合成数据恢复与覆盖率自检（独立于业务入口）：

    python -I tools/sim_check.py

运行模块检查：

    python -m unittest discover -s tests -v

检查源码编译：

    python -m compileall -q app tests tools
