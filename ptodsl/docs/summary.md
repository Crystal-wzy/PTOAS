# TileLang ST A5 Parallel Summary

- Total: 193
- Passed: 178
- Build failed: 10
- Run failed: 5

## Failed Build Testcases

| Name | Kind | Testcase | Phase |
| --- | --- | --- | --- |
| full_tcolexpanddiv | full | tcolexpanddiv | build |
| full_tcvt | full | tcvt | build |
| full_tdiv | full | tdiv | build |
| full_tinsert | full | tinsert | build |
| full_tinsert_acc2vec | full | tinsert_acc2vec | tilelangdsl fails too |
| full_tload | full | tload | build |
| full_tload_mat | full | tload_mat | tilelangdsl fails too |
| full_tmov | full | tmov | build |
| full_tstore_acc2gm | full | tstore_acc2gm | build |
| smoke_tdiv | smoke | tdiv | build |


## Failed Run Testcases

| Name | Kind | Testcase | Phase |
| --- | --- | --- | --- |
| full_tdivs | full | tdivs | compare |
| full_tfillpad | full | tfillpad | compare |
| full_trowargmax | full | trowargmax | compare |
| full_trowargmin | full | trowargmin | compare |
| full_tsort32 | full | tsort32 | compare |

## Passed Testcases

| Name | Kind | Testcase |
| --- | --- | --- |
| full_softmax | full | softmax |
| full_tabs | full | tabs |
| full_tadd | full | tadd |
| full_tadds | full | tadds |
| full_tand | full | tand |
| full_tands | full | tands |
| full_tcmp | full | tcmp |
| full_tcmps | full | tcmps |
| full_tcolargmax | full | tcolargmax |
| full_tcolargmin | full | tcolargmin |
| full_tcolexpand | full | tcolexpand |
| full_tcolexpandadd | full | tcolexpandadd |
| full_tcolexpandexpdif | full | tcolexpandexpdif |
| full_tcolexpandmax | full | tcolexpandmax |
| full_tcolexpandmin | full | tcolexpandmin |
| full_tcolexpandmul | full | tcolexpandmul |
| full_tcolexpandsub | full | tcolexpandsub |
| full_tcolmax | full | tcolmax |
| full_tcolmin | full | tcolmin |
| full_tcolprod | full | tcolprod |
| full_tcolsum | full | tcolsum |
| full_texp | full | texp |
| full_texpands | full | texpands |
| full_textract | full | textract |
| full_textract_fp | full | textract_fp |
| full_textract_v2v | full | textract_v2v |
| full_tfillpad_expand | full | tfillpad_expand |
| full_tfillpad_inplace | full | tfillpad_inplace |
| full_tfmod | full | tfmod |
| full_tfmods | full | tfmods |
| full_tgemv | full | tgemv |
| full_tinsert_vec | full | tinsert_vec |
| full_tlrelu | full | tlrelu |
| full_tlog | full | tlog | build |
| full_tmatmul | full | tmatmul |
| full_tmatmul_acc | full | tmatmul_acc |
| full_tmatmul_bias | full | tmatmul_bias |
| full_tmax | full | tmax |
| full_tmaxs | full | tmaxs |
| full_tmin | full | tmin |
| full_tmins | full | tmins |
| full_tmov2bias | full | tmov2bias |
| full_tmov2left | full | tmov2left |
| full_tmov2right | full | tmov2right |
| full_tmov2scale | full | tmov2scale |
| full_tmov2vec | full | tmov2vec |
| full_tmov_fp | full | tmov_fp |
| full_tmrgsort | full | tmrgsort |
| full_tmul | full | tmul |
| full_tmuls | full | tmuls |
| full_tneg | full | tneg |
| full_tnot | full | tnot |
| full_tor | full | tor |
| full_tors | full | tors |
| full_tpartadd | full | tpartadd |
| full_tpartmax | full | tpartmax |
| full_tpartmin | full | tpartmin |
| full_tpartmul | full | tpartmul |
| full_tprelu | full | tprelu |
| full_trandom | full | trandom |
| full_trecip | full | trecip |
| full_trelu | full | trelu |
| full_trem | full | trem |
| full_trems | full | trems |
| full_trowexpand | full | trowexpand |
| full_trowexpandadd | full | trowexpandadd |
| full_trowexpanddiv | full | trowexpanddiv |
| full_trowexpandexpdif | full | trowexpandexpdif |
| full_trowexpandmax | full | trowexpandmax |
| full_trowexpandmin | full | trowexpandmin |
| full_trowexpandmul | full | trowexpandmul |
| full_trowexpandsub | full | trowexpandsub |
| full_trowmax | full | trowmax |
| full_trowmin | full | trowmin |
| full_trowprod | full | trowprod |
| full_trowsum | full | trowsum |
| full_trsqrt | full | trsqrt |
| full_tsel | full | tsel |
| full_tsels | full | tsels |
| full_tshl | full | tshl |
| full_tshls | full | tshls |
| full_tshr | full | tshr |
| full_tshrs | full | tshrs |
| full_tsqrt | full | tsqrt |
| full_tsub | full | tsub |
| full_tsubs | full | tsubs |
| full_txor | full | txor |
| full_txors | full | txors |
| smoke_softmax | smoke | softmax |
| smoke_tabs | smoke | tabs |
| smoke_tadd | smoke | tadd |
| smoke_tadds | smoke | tadds |
| smoke_tand | smoke | tand |
| smoke_tands | smoke | tands |
| smoke_tcmp | smoke | tcmp |
| smoke_tcmps | smoke | tcmps |
| smoke_tcolargmax | smoke | tcolargmax |
| smoke_tcolargmin | smoke | tcolargmin |
| smoke_tcolexpand | smoke | tcolexpand |
| smoke_tcolexpandadd | smoke | tcolexpandadd |
| smoke_tcolexpanddiv | smoke | tcolexpanddiv |
| smoke_tcolexpandexpdif | smoke | tcolexpandexpdif |
| smoke_tcolexpandmax | smoke | tcolexpandmax |
| smoke_tcolexpandmin | smoke | tcolexpandmin |
| smoke_tcolexpandmul | smoke | tcolexpandmul |
| smoke_tcolexpandsub | smoke | tcolexpandsub |
| smoke_tcolmax | smoke | tcolmax |
| smoke_tcolmin | smoke | tcolmin |
| smoke_tcolprod | smoke | tcolprod |
| smoke_tcolsum | smoke | tcolsum |
| smoke_tcvt | smoke | tcvt |
| smoke_tdivs | smoke | tdivs |
| smoke_texp | smoke | texp |
| smoke_texpands | smoke | texpands |
| smoke_textract | smoke | textract |
| smoke_textract_fp | smoke | textract_fp |
| smoke_textract_v2v | smoke | textract_v2v |
| smoke_tfillpad | smoke | tfillpad |
| smoke_tfillpad_expand | smoke | tfillpad_expand |
| smoke_tfillpad_inplace | smoke | tfillpad_inplace |
| smoke_tfmod | smoke | tfmod |
| smoke_tfmods | smoke | tfmods |
| smoke_tload | smoke | tload |
| smoke_tlog | smoke | tlog | build |
| smoke_tlrelu | smoke | tlrelu |
| smoke_tmatmul | smoke | tmatmul |
| smoke_tmax | smoke | tmax |
| smoke_tmaxs | smoke | tmaxs |
| smoke_tmin | smoke | tmin |
| smoke_tmins | smoke | tmins |
| smoke_tmov | smoke | tmov |
| smoke_tmrgsort | smoke | tmrgsort |
| smoke_tmul | smoke | tmul |
| smoke_tmuls | smoke | tmuls |
| smoke_tneg | smoke | tneg |
| smoke_tnot | smoke | tnot |
| smoke_tor | smoke | tor |
| smoke_tors | smoke | tors |
| smoke_tpartadd | smoke | tpartadd |
| smoke_tpartmax | smoke | tpartmax |
| smoke_tpartmin | smoke | tpartmin |
| smoke_tpartmul | smoke | tpartmul |
| smoke_tprelu | smoke | tprelu |
| smoke_trandom | smoke | trandom |
| smoke_trecip | smoke | trecip |
| smoke_trelu | smoke | trelu |
| smoke_trem | smoke | trem |
| smoke_trems | smoke | trems |
| smoke_trowargmax | smoke | trowargmax |
| smoke_trowargmin | smoke | trowargmin |
| smoke_trowexpand | smoke | trowexpand |
| smoke_trowexpandadd | smoke | trowexpandadd |
| smoke_trowexpanddiv | smoke | trowexpanddiv |
| smoke_trowexpandexpdif | smoke | trowexpandexpdif |
| smoke_trowexpandmax | smoke | trowexpandmax |
| smoke_trowexpandmin | smoke | trowexpandmin |
| smoke_trowexpandmul | smoke | trowexpandmul |
| smoke_trowexpandsub | smoke | trowexpandsub |
| smoke_trowmax | smoke | trowmax |
| smoke_trowmin | smoke | trowmin |
| smoke_trowprod | smoke | trowprod |
| smoke_trowsum | smoke | trowsum |
| smoke_trsqrt | smoke | trsqrt |
| smoke_tsel | smoke | tsel |
| smoke_tsels | smoke | tsels |
| smoke_tshl | smoke | tshl |
| smoke_tshls | smoke | tshls |
| smoke_tshr | smoke | tshr |
| smoke_tshrs | smoke | tshrs |
| smoke_tsort32 | smoke | tsort32 |
| smoke_tsqrt | smoke | tsqrt |
| smoke_tsub | smoke | tsub |
| smoke_tsubs | smoke | tsubs |
| smoke_txor | smoke | txor |
| smoke_txors | smoke | txors |
