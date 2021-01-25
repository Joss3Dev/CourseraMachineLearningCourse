x = [];
for i = 1:500,
  if i < 10,
    filename = strcat('spam/0000',num2str(i),'.txt');
   else,
    if i < 100,
      filename = strcat('spam/000',num2str(i),'.txt');
     else,
      filename = strcat('spam/00',num2str(i),'.txt');
     endif
  endif
  file_contents = readFile(filename);
  word_indices  = processEmail(file_contents);
  x             = [x emailFeatures(word_indices)];
endfor

for i = 1:250,
  if i < 10,
    filename = strcat('easy_ham/0000',num2str(i),'.txt');
   else,
    if i < 100,
      filename = strcat('easy_ham/000',num2str(i),'.txt');
     else,
      filename = strcat('easy_ham/00',num2str(i),'.txt');
     endif
  endif
  file_contents = readFile(filename);
  word_indices  = processEmail(file_contents);
  x             = [x emailFeatures(word_indices)];
endfor


for i = 1:250,
  if i < 10,
    filename = strcat('hard_ham/0000',num2str(i),'.txt');
   else,
    if i < 100,
      filename = strcat('hard_ham/000',num2str(i),'.txt');
     else,
      filename = strcat('hard_ham/00',num2str(i),'.txt');
     endif
  endif
  file_contents = readFile(filename);
  word_indices  = processEmail(file_contents);
  x             = [x emailFeatures(word_indices)];
endfor

