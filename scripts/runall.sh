mkdir data
mkdir log
for t in jetengine vanderpol quadrotor_C3M f16_GCAS_sphInit f16_GCAS
do
	python3 NeuReach.py --no_cuda --system $t --log log/log_${t}_ours/ --data_file_train data/${t}_traces_train.pklz --data_file_eval data/${t}_traces_eval.pklz 1>/dev/null 2>&1
	python3 DryVR.py --no_cuda --system $t --log log/log_${t}_dryvr/ --data_file_train data/${t}_traces_train.pklz --data_file_eval data/${t}_traces_eval.pklz 1>/dev/null 2>&1
done

for t in jetengine vanderpol quadrotor_C3M f16_GCAS_sphInit f16_GCAS
do
	echo $t
	scripts/test.sh $t log/log_${t}_ours
	scripts/test.sh $t log/log_${t}_dryvr
	echo "----------------"
done

rm test.txt
