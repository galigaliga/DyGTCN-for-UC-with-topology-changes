clc
clear
close all

% 载入数据
gen_data = readmatrix('./ieee118new/gen.csv');
bus_data = readmatrix('./ieee118new/bus.csv');
branch_data = readmatrix('./ieee118new/branch.csv');
load_data = readmatrix('./ieee118new/new_busloads.csv'); 
price_data = readmatrix('./ieee118new/price.csv');

% 常量参数
T_num = 24;
time = 365;

% 机组、母线、边数量
gen_num = size(gen_data, 1);
bus_num = size(bus_data, 1);
branch_num = size(branch_data, 1);

% 机组参数
gen_bus = gen_data(:, 2);
PG_min = gen_data(:, 3);
PG_max = gen_data(:, 4);
R_up = gen_data(:, 5);
R_down = gen_data(:, 6);
TG_off = gen_data(:, 7);
TG_on = gen_data(:, 8);


% 支路数据
m = branch_data(:, 2);
n = branch_data(:, 3);
R = branch_data(:, 5);
X = branch_data(:, 6);
B = branch_data(:, 7);
PL_max = branch_data(:, 8);
PL_min = -branch_data(:, 8);
ratio = branch_data(:, 9);
angle = branch_data(:, 10);
stat = branch_data(:, 11);

% 成本数据
a = price_data(:, 1);
b = price_data(:, 2);
c = price_data(:, 3);

% 计算ptdf
ptdf = readmatrix('./ieee118new/ptdf.csv');

% 将发电机出力扩展到所有节点，非发电机节点出力为0
i = gen_bus;
j = 1:1:gen_num;
v = ones(1, gen_num);
size1 = bus_num;
size2 = gen_num;
nC = sparse(i, j, v, size1, size2);
C = kron(eye(T_num), nC);

cal_load = bus_data(:, 2);
rate = cal_load./ sum(cal_load);

disp('完成数据读取')


% 机组分类
big_gens = find(PG_max >= 300);    % 大机组编号
small_gens = find(PG_max < 300);   % 小机组编号
big_gen_num = length(big_gens);
small_gen_num = length(small_gens);
% 初始化检修记录表（全局变量）
global maintenance_schedule;
global last_maintained_week;
maintenance_schedule = struct('start_day', {}, 'gens', {});
last_maintained_week = zeros(gen_num, 1);

% 设定起始与结束日期
power = [];
state = [];
startday = 1;
endday = 365;
round = 1;
filename_power = 'power.csv';
filename_state = 'state.csv';
filename_relaxflow='relaxflow.txt';

% 初始化检修记录表和最优成本记录表
maintenance_data = zeros(2, 365);
daily_cost = -ones(1,365);

for k = startday:endday
   % 每日初始化当前检修机组
    current_maintenance_gens = [];
    % 每日清理过期维护任务
    global maintenance_schedule;
    valid_idx = []; % 保存未过期任务的索引
    for idx = 1:length(maintenance_schedule)
        if maintenance_schedule(idx).start_day +7 > k % 任务未过期
            valid_idx = [valid_idx, idx];
        end
    end
    maintenance_schedule = maintenance_schedule(valid_idx); % 保留有效任务
    
    % --- 大机组每3周调度 ---
    if mod(k, 21) == 1
        % 获取符合条件的大机组（初始未维护 + 满足间隔）
        is_initial = (last_maintained_week(big_gens) == 0);
        is_interval_met = (k - last_maintained_week(big_gens) >= 21);
        eligible_big = big_gens(is_initial | is_interval_met);
        
        if ~isempty(eligible_big)
            % 创建候选机组队列（优先初始未维护机组）
            initial_candidates = eligible_big(last_maintained_week(eligible_big) == 0);
            non_initial_candidates = eligible_big(last_maintained_week(eligible_big) ~= 0);
            
            % 随机打乱初始机组顺序，按维护时间排序非初始机组
            initial_candidates = initial_candidates(randperm(length(initial_candidates)));
            [~, sort_idx] = sort(last_maintained_week(non_initial_candidates));
            sorted_non_initial = non_initial_candidates(sort_idx);
            
            % 合并候选列表
            candidates = [initial_candidates; sorted_non_initial];
            found_feasible = false;
            
            % 遍历所有候选机组
            for i = 1:length(candidates)
                current_candidate = candidates(i);
                
                % 冲突检查
                if check_conflict(current_candidate, k)
                    feasible = true;
                    % 容量可行性检查
                    for d = k:min(k+6, endday)
                        existing_gens = get_current_maintenance(d);
                        all_gens = union(existing_gens, current_candidate);
                        required = max(load_data(:, d));
                        available = sum(PG_max) - sum(PG_max(all_gens));
                        
                        if available < required
                            feasible = false;
                            disp(['候选机组 ',num2str(current_candidate),' 第',num2str(d),'天容量不足']);
                            break;
                        end
                    end
                    
                    % 找到可行机组
                    if feasible
                        maintenance_schedule(end+1) = struct('start_day',k, 'gens',current_candidate);
                        last_maintained_week(current_candidate) = k;
                        found_feasible = true;
                        break; % 跳出循环
                    end
                end
            end
            
            % 所有候选机组均不可行
            if ~found_feasible
                disp(['第',num2str(k),'天所有候选大机组均无法满足容量要求']);
            end
        end
    end

    
    % --- 小机组调度（大机组间隔期）---
    if mod(k,21) == 8 || mod(k,21) == 15
        is_small_initial = (last_maintained_week(small_gens) == 0);
        is_small_interval_met = (k - last_maintained_week(small_gens) >= 21);
        eligible_small = small_gens(is_small_initial | is_small_interval_met);
        
        if ~isempty(eligible_small)
            % 优先排序：初始未维护在前，非初始按上次维护时间升序排列
            initial_small = eligible_small(last_maintained_week(eligible_small) == 0);
            non_initial_small = eligible_small(last_maintained_week(eligible_small) ~= 0);
            [~, sort_idx] = sort(last_maintained_week(non_initial_small), 'ascend');
            initial_small = initial_small(randperm(length(initial_small)));
            candidates_small = [initial_small; non_initial_small(sort_idx)];
            
            % 尝试选择两个机组
            feasible = false;
            if length(candidates_small) >= 2
                selected_gens = candidates_small(1:2); % 优先选前两个
                % 检查容量可行性
                feasible = true;
                for d = k:min(k+6, endday)
                    existing_gens = get_current_maintenance(d);
                    all_gens = union(existing_gens, selected_gens);
                    required = max(load_data(:, d));
                    available = sum(PG_max) - sum(PG_max(all_gens));
                    if available < required
                        feasible = false;
                        break;
                    end
                end
                if feasible
                    maintenance_schedule(end+1) = struct('start_day',k, 'gens',selected_gens);
                    last_maintained_week(selected_gens) = k;
                end
            end
            
            % 若选两个不可行，尝试选单个
            if ~feasible
                for i = 1:length(candidates_small)
                    current_gen = candidates_small(i);
                    feasible_single = true;
                    for d = k:min(k+6, endday)
                        existing_gens = get_current_maintenance(d);
                        all_gens = union(existing_gens, current_gen);
                        required = max(load_data(:, d));
                        available = sum(PG_max) - sum(PG_max(all_gens));
                        if available < required
                            feasible_single = false;
                            break;
                        end
                    end
                    if feasible_single
                        maintenance_schedule(end+1) = struct('start_day',k, 'gens',current_gen);
                        last_maintained_week(current_gen) = k;
                        feasible = true;
                        break;
                    end
                end
                if ~feasible
                    disp(['第', num2str(k), '天无法安排小机组，容量不足']);
                end
            end
        end
    end
    
    % 获取当天检修机组
    current_maintenance_gens = get_current_maintenance(k);

    % 存储当天检修机组编号（转为字符串）
    current_maintenance = current_maintenance_gens; % 替换为实际获取当天检修机组的代码（得到一个数组）
    sorted_gens = sort(current_maintenance(:))';
    if ~isempty(sorted_gens)
        % 填入对应列
        maintenance_data(1:min(2, length(sorted_gens)), k) = sorted_gens(1:min(2, end));
    end

    % 机组状态参数
    u = binvar(gen_num, T_num, 'full');%机组状态变量
    P = sdpvar(gen_num, T_num, 'full');%机组总出力
    v = binvar(gen_num, T_num, 'full');%机组开启动作指令，启动为1，关停为0
    w = binvar(gen_num, T_num, 'full'); % 机组关停动作指令，关停为1，启动为0
    % 添加潮流约束松弛项
    relax_flow = sdpvar(branch_num, T_num);
    % 潮流约束松弛惩罚
    penalty_flow_relax = 0;

    % 负荷参数
    total_load = load_data(:, k);
    bus_load = rate * total_load';

    % 负荷侧调整变量及相关成本参数
    demand_load = sdpvar(T_num, 1); % 定义每个时刻需要调整（减少）的负荷量，作为变量
    cost_per_load_reduction = 150; % 每减少一单位负荷花费的成本，可按需修改
    demand_reduction_cost = 0; % 初始化负荷减少成本

    cost = 0;
    tic % 计时开始
    for t = 1:T_num
        for i = 1:gen_num
            cost = cost + a(i) * P(i, t) * P(i, t) + b(i) * P(i, t) + c(i);
            % 计算每个时刻因负荷调整产生的成本并累加到总成本
            demand_reduction_cost = demand_reduction_cost + cost_per_load_reduction * demand_load(t);
        end
    end
    cost = cost + demand_reduction_cost; % 将负荷调整成本加到总成本中
    disp('成本构建')

    % 约束条件初始化
    st = [];
    % 当前需检修机组日内停机约束
    if ~isempty(current_maintenance_gens)
        for t = 1:T_num
            st = [st, u(current_maintenance_gens, t) == 0]; % 强制停机
        end
    end
    disp('机组检修停机约束')
    % 机组出力上下限约束
    for t = 1:T_num  
        for i = 1:gen_num  
            st = [st, u(i, t) * PG_min(i) <= P(i, t) <= u(i, t) * PG_max(i)];   
        end  
    end
    disp('机组出力上下限约束')
    % 启停状态逻辑约束
    for t = 2:T_num
        for i = 1:gen_num
            st = [st, u(i, t) - u(i, t-1) == v(i, t) - w(i, t)];
        end
    end
    disp('启停状态逻辑约束')
    % 机组爬坡约束
    for t = 2:T_num  
        for i = 1:gen_num   
            st = [st, P(i,t) - P(i,t-1) <= R_up(i)*u(i,t-1) + PG_min(i)*v(i,t)];  
            st = [st, P(i, t-1) - P(i, t) <= R_down(i)*u(i, t-1) + PG_min(i)*w(i, t)];
        end  
    end  
    % 启动时间约束
    for i = 1:gen_num
        for t = TG_on(i):T_num
            st = [st, sum(v(i,[t-TG_on(i)+1:t])) <= u(i,t)];
        end
    end
    % 关停时间约束
    for i = 1:gen_num
        for t = TG_off(i):T_num
            st = [st, sum(w(i,[t-TG_off(i)+1:t])) <= 1 - u(i,t)];
        end
    end  
    % 启停次数约束
    for i = 1:gen_num
        st = [st, sum(v(i,:)) + sum(w(i,:)) <= 1];
    end
    disp('启停时间约束')
    % 负荷调整约束
    for t = 1:T_num
        % 负荷调整量约束分时段调节
        if (k > 180 && k < 270) % 夏季高需求期
            st = [st, 0 <= demand_load(t) <= 0.15 * total_load(t)]; % 允许15%负荷调整
        else
            st = [st, 0 <= demand_load(t) <= 0.10 * total_load(t)]; % 允许10%负荷调整
        end
    end
    disp('负荷调整约束')
    % 潮流约束
    for t = 1:T_num
        % 潮流约束计算中减去负荷调整负荷
        st = [st, PL_min - relax_flow(:, t) <= ptdf * (nC * P(:, t) - (bus_load(:, t) - rate * demand_load(t))) <= PL_max + relax_flow(:, t)];
        st = [st, relax_flow(:, t) >= 0];  % 确保松弛项不为负
    end
    disp('潮流约束（考虑负荷调整后）')
    % 负载平衡约束
    for t = 1:T_num
        st = [st, sum(P(:,t)) == total_load(t) - demand_load(t)];
    end
    disp('负载平衡约束')
    % 目标函数添加惩罚
    penalty_flow_cost = 30000;  % 潮流约束松弛的惩罚系数
    % 计算惩罚
    penalty_flow_relax = penalty_flow_cost * sum(relax_flow(:));
    % 最终目标函数
    cost = cost + penalty_flow_relax  ;

    disp('完成约束构建')

    % gurobi求解
    ops = sdpsettings('solver','gurobi', 'verbose',2, 'gurobi.MIPGap', 0.01, 'gurobi.Heuristics', 0.5);  % 采用Gurobi求解器
    result = solvesdp(st,cost);

    % 计时结束
    toc
    disp(['天数：',num2str(value(k)),'最小经济成本为：',num2str(value(cost)),'$']);
    
    %每日成本记录
    current_cost = value(cost); % 提取当前成本值
    daily_cost(k) = current_cost; % 存储到对应天数

    if value(cost)>0
        P = value(P);
        P = reshape(P, [], 1);
        u = value(u);
        u = reshape(u,[],1);
    else 
        P = -1 * ones(54 * 24, 1);  % 修改为全 -1
        u = -1 * ones(54 * 24, 1);  % 修改为全 -1
    end

    % 创建带有列名的表格
    tbl_power = array2table(P, 'VariableNames', {['instance', num2str(k)]});
    tbl_state = array2table(u, 'VariableNames', {['instance', num2str(k)]});

    if round == 1
        % 第一次写入，包括列名
        writetable(tbl_power, filename_power);
        writetable(tbl_state, filename_state);
    else
        % 读取现有CSV文件，合并新数据，并写回文件
        existing_power = readtable(filename_power);
        existing_state = readtable(filename_state);

        % 将新数据作为新列追加到现有表格
        existing_power = [existing_power, tbl_power];
        existing_state = [existing_state, tbl_state];

        % 写回CSV文件
        writetable(existing_power, filename_power);
        writetable(existing_state, filename_state);
    end

    round = round + 1;


    if value(cost)>0
        relax_flow = value(relax_flow);
        [row, col, val] = find(relax_flow);  % 找到所有非零值的行号、列号和值
        output_file = filename_relaxflow;
        % 打开文件（'a'模式表示追加写入）
        fid = fopen(output_file, 'a');

        if fid == -1
            error('无法打开文件');
        end

        % 写入第k天结果的标题
        fprintf(fid, '第 %d 天结果:\n', k);
        % 输出结果
        for fc = 1:24
            fprintf(fid, 'Column %d:\n', fc);
            % 查找当前列中非零值的行号和值
            rows_in_col = row(col == fc);  % 当前列非零值的行号
            values_in_col = val(col == fc);  % 当前列非零值

            % 输出行号和值
            for k = 1:length(rows_in_col)
                fprintf(fid, 'Row %d, Value: %.2f\n', rows_in_col(k), values_in_col(k));
            end
            fprintf(fid, '\n');  % 分隔每一列的输出
        end

        % 关闭文件
        fclose(fid);

        disp('潮流松弛结果已保存到txt文件中');

    else
        disp('该天无解');
    end

end

function current_gens = get_current_maintenance(current_day)
    current_gens = [];
    global maintenance_schedule;
    for idx = 1:length(maintenance_schedule)
        start_day = maintenance_schedule(idx).start_day;
        if current_day >= start_day && current_day < start_day +7 % 修改为 <
            current_gens = union(current_gens, maintenance_schedule(idx).gens);
        end
    end
end


function valid = check_conflict(selected_gens, current_week)
    global last_maintained_week;
    min_interval = 21;
    % 初始值为0时直接允许调度
    valid = all(last_maintained_week(selected_gens) == 0 | ...
               (current_week - last_maintained_week(selected_gens) >= min_interval));
end


%存机组检修表
fid = fopen('maintenance.csv','w');
fprintf(fid, 'Day%d,', 1:364); % 前364列表头
fprintf(fid, 'Day365\n');       % 第365列表头
% 第一行数据（若有多个机组只取第一个）
fprintf(fid, '%d,', maintenance_data(1,1:end-1)); 
fprintf(fid, '%d\n', maintenance_data(1,end)); 
% 第二行数据（第二个机位）
fprintf(fid, '%d,', maintenance_data(2,1:end-1)); 
fprintf(fid, '%d\n', maintenance_data(2,end)); 
fclose(fid);

%存最优成本表
csv_data = num2cell(daily_cost);
header = compose('Day%d',1:365);
cost_table = cell2table(csv_data, 'VariableNames', header); 
writetable(cost_table, 'cost.csv'); 