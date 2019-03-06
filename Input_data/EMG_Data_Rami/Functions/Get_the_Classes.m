function noisy_file=Get_the_Classes(Gesture)


noisy_file=[];

for k=1:size(Gesture,1)
    
    noisy_file=strcat(noisy_file,num2str(Gesture(k)),'_');
end
d=1;