

cfg = [];
cfg.dataset = "/home/dpedrosac/Schreibtisch/rsEEG/rsEEG_15/DBS_15_1.eeg";
data = ft_preprocessing(cfg);


%%
x = data.trial{1};
sr = data.fsample;

DATAlength=size(x,2);

%) parity check
if((DATAlength/2)~=round(DATAlength/2))
    x=x(:,1:end-1);
    FFTlength=DATAlength-1;
else
    FFTlength=DATAlength;
end

%) process fft
fprintf('DBSFILT >> Process fft...  ')
Y = fft(x,[],2);
Y=2*abs(Y);
f = sr/2*linspace(0,1,FFTlength/2+1);
fprintf('Done.\n')

%)split
if(isvector(Y))
    Ym=Y;
else
    Ym=mean(Y);
end
Ym1=Ym(1:(FFTlength/2+1));

%)prepare Spikes matrix
Spikes=zeros(8,length(f));
Spikes(1,:)=f;
Spikes(2,:)=Ym1;

%% Spikes detection

type    = 2; % Hampel identifier and refined spike identification
HampelL = 1; % windows size for aut. spike detection (Hz)
HampelT = 2; % Hampel threshold for automatic spike detection.
nmax = 5;
eps=.01;



Y=Spikes(2,:)'; % Mean spectrum
Fres=Spikes(1,2); % frequency resolution
WL=round(HampelL./Fres); % windows length in the frequency space
nbW=floor(length(Y)/WL); % number of full length windows
L=nbW*WL; % length of reduced data

Ya=Y(1:L);
Yb=Y((end-(WL-1)):end);

%) epoch the data
Ya=reshape(Ya,WL,nbW);
Ya=Ya';
%) calculate the Hampel identifier for each epoch
YaMedian=median(Ya,2);
YaThres=HampelT.*1.4286.*YaMedian;
%) Transform to a linear threshold
YaMedianLin=kron(ones(1,WL),YaMedian);
YaThresLin=kron(ones(1,WL),YaThres);
YaMedianLin=reshape(YaMedianLin',[],1);
YaThresLin=reshape(YaThresLin',[],1);

%) Last epoch processing
YbMedian=median(Yb);
YbThres=HampelT.*1.4286.*YbMedian;
YbMedianLin=kron(ones(1,WL),YbMedian);
YbThresLin=kron(ones(1,WL),YbThres);

%) merge epochs
Ymedian=0.*Y;
Ythres=0.*Y;
Ymedian(1:L)=YaMedianLin;
Ythres(1:L)=YaThresLin;
Ymedian((end-(WL-1)):end)=YbMedianLin;
Ythres((end-(WL-1)):end)=YbThresLin;

Spikes(3,:)=Ymedian;
Spikes(4,:)=Ythres;
Spikes(5,:)=Y>Ythres;
Spikes(6,:)=Y>Ythres;

nb_spikes=sum(Spikes(5,:));

if(type==2) % launch Spikes Identification
    
    SpikesIndex=find(Spikes(5,:)==1);
    for spk=1:nb_spikes
        
        Fs=Spikes(1,SpikesIndex(spk));
        if (Fs > FdbsL-.6 && Fs < FdbsL+.5)
            %keyboard
        end
        [dbs_induced,n,h]=DBSFILT_testspike(Fs,FdbsL,FdbsR,nmax,eps,0);
        
        Spikes(6,SpikesIndex(spk))=dbs_induced;
        Spikes(7,SpikesIndex(spk))=n;
        Spikes(8,SpikesIndex(spk))=h;
        
    end
    
    nb_spikes=sum(Spikes(6,:));
    
end


%% Spikes Removal

x = data.trial{1};
DATAlength=size(x,2);

%) parity check 
if((DATAlength/2)~=round(DATAlength/2))
    x=x(:,1:end-1);
    FFTlength=DATAlength-1;
else
    FFTlength=DATAlength;
end

%) step 2 - transpose data to the frequency space (process fft).
fprintf('DBSFILT >> Process fft...  ')
Y = fft(x,[],2);
f = sr/2*linspace(0,1,FFTlength/2+1);
fprintf('Done.\n');

%)split
Y1=Y(:,1:(FFTlength/2+1));
Y=fliplr(Y);
Y2=0.*Y1;
Y2(:,2:end)=Y(:,1:(FFTlength/2));
Y2(:,1)=Y1(:,1);

%)step 3 - find detected spikes, and spike interpolation

Fdbs=find(spikes(6,:)>0);
nb_spikes=length(Fdbs);

Fres=f(2);
Wl=1;
Wls=round(((Wl/Fres)-1)./2);

for i=1:nb_spikes
    if((Fdbs(i)-Wls)<=0)
        Y1w=Y1(:,1:Fdbs(i)+Wls);
        Y2w=Y2(:,1:Fdbs(i)+Wls);
    elseif((Fdbs(i)+Wls)>length(Y1))
        Y1w=Y1(:,Fdbs(i)-Wls:end);
        Y2w=Y2(:,Fdbs(i)-Wls:end);
    else
        Y1w=Y1(:,Fdbs(i)-Wls:Fdbs(i)+Wls);
        Y2w=Y2(:,Fdbs(i)-Wls:Fdbs(i)+Wls);
    end
    
    Y1med=median(Y1w,2);
    Y1(:,Fdbs(i))=Y1med;
    
    Y2med=median(Y2w,2);
    Y2(:,Fdbs(i))=Y2med;
    
end

%)step 4 - Signal reconstruction
Y2=fliplr(Y2);
Y=[Y1(:,1:end-1),Y2(:,1:end-1)];
x = ifft(Y,[],2,'symmetric');

if(DATAlength>FFTlength)
    lastsample=x(:,end)+(x(:,end)-x(:,end-1)); %linear interpolation of the last sample, if necessary.
    x(:,end+1)=lastsample;
end
