function [Subject, Gesture, Trial]=Extract_trial_info(Trial)

Word=char(Trial);
semiC=strfind(Word,'-');
fin=strfind(Word,'.mat');

Subject=str2double(Word(+1:semiC(1)-1));
Gesture=str2double(Word(semiC(1)+1:semiC(2)-1));
Trial=str2double(Word(semiC(2)+1:fin-1));


end