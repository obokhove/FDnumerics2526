clear; clc; close all;
% solution 2(b)

alpha = 0.4709;
beta  = 1.0;
H  = 1.0;
bB = 1.178164343;
bT = 0.585373798;

times = [0.05 0.1 0.2 0.5 1.0 2.0];
gridSizes = [11 21 41];

for Nz = gridSizes
    
    z = linspace(0,H,Nz)';
    dz = z(2)-z(1);
    
    % Initial condition
    bt = bT*ones(Nz,1);
    bt(1)=bB;
    bt(end)=bT;
    
    dt = 1e-5;
    tmax = max(times);
    
    Bstore = zeros(Nz,length(times));
    saved = false(size(times));
    
    t = 0;
    k=1;
    
    while t < tmax
        
        b3 = bt.^3;
        bt_new = bt;
        
        for j = 2:Nz-1
            
            % Convective term
            con = alpha*(b3(j)-b3(j-1))/dz;
            
            % Diffusive term
            dif = beta*(((b3(j+1)+b3(j))/2)*(bt(j+1)-bt(j)) - ((b3(j)+b3(j-1))/2)*(bt(j)-bt(j-1)) )/dz^2;
            
            bt_new(j) = bt(j) - dt*con + dt*dif;
        end
        
        bt_new(1)=bB;
        bt_new(end)=bT;
        
        bt = bt_new;
        t = t + dt;

        if k <= length(times) && t >= times(k)
           Bstore(:,k) = bt;
           k = k + 1;
        end

    end
    
    % Plots
    figure;
    hold on
    
    for k=1:length(times)
        plot(Bstore(:,k),z,'LineWidth',1.5)
        hold on
        plot(-Bstore(:,k),z,'LineWidth',1.5)
    end
    
  
    xlabel('dike width')
    ylabel('z')
    title(['grid point Nz = ' num2str(Nz)])
    legend('t=0.05','t=0.1','t=0.2','t=0.5','t=1','t=2','Location','best')
    grid on
    hold off
    
end
