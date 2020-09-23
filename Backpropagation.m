clc; clear; close all; warning off all;

%data yang akan diuji dan dilatih
data = xlsread('data.xlsx', 'data');
sort=1:18;
datalatih=data(sort(1:15),:);
datatesting=data(sort(16:end),:);

%mengubah kolom jadi baris
latih=datalatih(:,2:end)';
target=datalatih(:,1)';
ujidata=datatesting(:,2:end)';

targetujidata=datatesting(:,1)'; 
[~,N] = size(latih);

%Pembuatan Jaringan Backpropagation
net = newff(minmax(latih),[10 1],{'logsig','purelin'},'traingdx');
%Inisialisasi bobot & bias
net.IW{1,1} = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0;
               0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0];
net.LW{2,1} = [0,0,0,0,0,0,0,0,0,0];
net.b{1,1} = [0;0;0;0;0;0;0;0;0;0];
net.b{2,1} = [0];
net.performFcn = 'mse';
net.trainParam.goal = 0.01;
net.trainParam.epochs = 200;
net.trainParam.lr = 0.1;

%proses pelatihan dengan 15 data latih
[net_keluaran,tr,Y,E] = train(net,latih,target);

bobot_hidden = net_keluaran.IW{1,1};
bobot_keluaran = net_keluaran.LW{2,1};
bias_hidden = net_keluaran.b{1,1};
bias_keluaran = net_keluaran.b{2,1};
jumlah_iterasi = tr.num_epochs;
nilai_keluaran = Y;
nilai_error = E;
error_MSE = (1/N)*sum(nilai_error.^2);

%proses pengujian dengan 3 data uji
hasil_uji=round(sim(net_keluaran,ujidata));

%menampilkan hasil uji
disp('Data ke- Target Hasil')
disp([(1:length(targetujidata))' targetujidata' hasil_uji'])
