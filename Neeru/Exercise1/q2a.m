clear; clc; close all;

% steady state (A)

alpha = 0.4709;
beta = 1.0;
Q = 0.99;
H = 1.0;
bB = 1.178164343;
Nz = 2000;

z = linspace(0,H,Nz)';
dz = z(2)-z(1);

b = zeros(Nz,1);
b(1)=bB;

for i=1:Nz-1
    dbdz = alpha/beta - Q/(beta*b(i)^3);
    b(i+1) = b(i) + dz*dbdz;
end

% Steady-state profile

figure;
plot(b,z,'LineWidth',2)

xlabel('b(z)')
ylabel('z')
title('Steady-state solution')
grid on

% Cross-section plot

figure;
plot(b/2,z,'k','LineWidth',2)
hold on
plot(-b/2,z,'k','LineWidth',2)
xlabel('Dike width')
ylabel('z')
title('Cross section')
grid on


