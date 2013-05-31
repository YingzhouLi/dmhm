%%
filename='./ALocal_';
A=[];
for i=0:7
    run([filename int2str(i) '.m']);
    A=[A;ALocal{i+1}];
end

imagesc(A);

% %%
% filename='./BLocal_';
% B=[];
% for i=0:7
%     run([filename int2str(i) '.m']);
%     B=[B;BLocal{i+1}];
% end
% 
% imagesc(B);

% %%
% filename='./ELocal_';
% E=[];
% for i=0:7
%     run([filename int2str(i) '.m']);
%     E=[E;ELocal{i+1}];
% end
% T=-log10(abs(E));
% imagesc(-log10(abs(E)));
% 
% %%
% filename='./YLocal_';
% Y=[];
% for i=0:7
%     run([filename int2str(i) '.m']);
%     Y=[Y;YLocal{i+1}];
% end
% 
% imagesc(Y);

% %%
% filename='./zlocal_';
% z=[];
% for i=0:0
%     run([filename int2str(i) '.m']);
%     z=[z;zlocal{i+1}];
% end
% 
% %imagesc(z);
% y=a*a;
% e=abs(y-z);
% imagesc(e);

%%
filename='./ELocal_';
E=[];
for i=0:0
    run([filename int2str(i) '.m']);
    E=[E;ELocal{i+1}];
end
imagesc(abs(E));

%%
filename='./YLocal_';
Y=[];
for i=0:3
    run([filename int2str(i) '.m']);
    Y=[Y;YLocal{i+1}];
end
imagesc(abs(Y));

%%
filename='./ZLocal_';
Z=[];
for i=0:7
    run([filename int2str(i) '.m']);
    Z=[Z;ZLocal{i+1}];
end
imagesc(abs(Z))

%%
run(['./E.m']);
E=EMat;
imagesc(abs(E));