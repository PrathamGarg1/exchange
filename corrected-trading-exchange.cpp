#include <iostream>
#include <map>
#include <set>
#include <queue>
#include <unordered_map>
#include <vector>
#include <string>
#include <algorithm>
#include <iomanip>
#include <limits>
#include <stack>
#include <chrono>
#include <random>
#include <cassert>
#include <deque>
#include <cmath>

using namespace std;

class Order;
class Portfolio;
class Trade;
class AssetGraph;
class PortfolioOptimizer;

// Enums
enum class OrderSide { BUY, SELL };
enum class OrderStatus { PENDING, FILLED, PARTIALLY_FILLED, CANCELLED };
enum class OrderType { MARKET, LIMIT };

// Order class
class Order {
public:
    static int nextOrderId;
    int id;
    string symbol;
    OrderSide side;
    OrderType type;
    double price;
    int quantity;
    int filledQuantity;
    OrderStatus status;
    string trader;
    chrono::system_clock::time_point timestamp;

    Order(string sym, OrderSide s, OrderType t, double p, int q, string tr)
        : id(++nextOrderId), symbol(sym), side(s), type(t), price(p), quantity(q), 
          filledQuantity(0), status(OrderStatus::PENDING), trader(tr),
          timestamp(chrono::system_clock::now()) {}

    int getRemainingQuantity() const { return quantity - filledQuantity; }
    bool isFullyFilled() const { return filledQuantity >= quantity; }
};

int Order::nextOrderId = 0;

// Trade execution record
class Trade {
public:
    static int nextTradeId;
    int id;
    string symbol;
    double price;
    int quantity;
    string buyer, seller;
    chrono::system_clock::time_point timestamp;

    Trade(string sym, double p, int q, string b, string s)
        : id(++nextTradeId), symbol(sym), price(p), quantity(q), buyer(b), seller(s),
          timestamp(chrono::system_clock::now()) {}
};

int Trade::nextTradeId = 0;

// Asset price history for GNN-like prediction
class AssetPriceHistory {
public:
    string symbol;
    deque<double> prices;
    deque<double> returns;
    static const int MAX_HISTORY = 100;
    
    // DEFAULT CONSTRUCTOR ADDED - FIX #1
    AssetPriceHistory() : symbol("") {}
    
    AssetPriceHistory(string sym) : symbol(sym) {}
    
    void addPrice(double price) {
        prices.push_back(price);
        if (prices.size() > 1) {
            double ret = log(price / prices[prices.size()-2]);
            returns.push_back(ret);
            if (returns.size() > MAX_HISTORY) returns.pop_front();
        }
        if (prices.size() > MAX_HISTORY) prices.pop_front();
    }
    
    double getCurrentPrice() const {
        return prices.empty() ? 0.0 : prices.back();
    }
    
    double getVolatility() const {
        if (returns.size() < 2) return 0.01;
        double mean = 0.0;
        for (double ret : returns) mean += ret;
        mean /= returns.size();
        
        double variance = 0.0;
        for (double ret : returns) {
            variance += (ret - mean) * (ret - mean);
        }
        return sqrt(variance / (returns.size() - 1));
    }
    
    vector<double> getRecentReturns(int count = 10) const {
        vector<double> recent;
        int start = max(0, (int)returns.size() - count);
        for (int i = start; i < returns.size(); i++) {
            recent.push_back(returns[i]);
        }
        return recent;
    }
};

// Graph representation for financial assets
class AssetGraph {
private:
    map<string, int> symbolToIndex;
    map<int, string> indexToSymbol;
    vector<vector<double>> transactionCosts;
    vector<vector<double>> correlations;
    vector<vector<pair<int, double>>> adj;
    map<string, AssetPriceHistory> priceHistory;
    int numAssets;
    
public:
    AssetGraph() : numAssets(0) {}
    
    void addAsset(const string& symbol) {
        if (symbolToIndex.find(symbol) != symbolToIndex.end()) return;
        
        symbolToIndex[symbol] = numAssets;
        indexToSymbol[numAssets] = symbol;
        
        // FIX #2: Use emplace instead of operator[]
        priceHistory.emplace(symbol, AssetPriceHistory(symbol));
        
        numAssets++;
        
        // Resize matrices
        transactionCosts.resize(numAssets, vector<double>(numAssets, 0.0));
        correlations.resize(numAssets, vector<double>(numAssets, 0.0));
        adj.resize(numAssets);
        
        // Initialize diagonal elements
        int idx = numAssets - 1;
        transactionCosts[idx][idx] = 0.0;
        correlations[idx][idx] = 1.0;
    }
    
    void updatePrice(const string& symbol, double price) {
        if (symbolToIndex.find(symbol) == symbolToIndex.end()) {
            addAsset(symbol);
        }
        priceHistory[symbol].addPrice(price);
        updateTransactionCosts();
        updateCorrelations();
    }
    
    // GNN-inspired transaction cost prediction
    void updateTransactionCosts() {
        for (int i = 0; i < numAssets; i++) {
            for (int j = 0; j < numAssets; j++) {
                if (i == j) {
                    transactionCosts[i][j] = 0.0;
                } else {
                    string sym1 = indexToSymbol[i];
                    string sym2 = indexToSymbol[j];
                    
                    double vol1 = priceHistory[sym1].getVolatility();
                    double vol2 = priceHistory[sym2].getVolatility();
                    double price1 = priceHistory[sym1].getCurrentPrice();
                    double price2 = priceHistory[sym2].getCurrentPrice();
                    
                    // Transaction cost based on volatility and price difference
                    double baseCost = 0.001; // 0.1% base cost
                    double volatilityCost = (vol1 + vol2) * 0.5;
                    double priceDiffCost = abs(log(price1) - log(price2)) * 0.01;
                    
                    transactionCosts[i][j] = baseCost + volatilityCost + priceDiffCost;
                }
            }
        }
    }
    
    // Calculate correlations for MST construction
    void updateCorrelations() {
        for (int i = 0; i < numAssets; i++) {
            for (int j = 0; j < numAssets; j++) {
                if (i == j) {
                    correlations[i][j] = 1.0;
                } else {
                    string sym1 = indexToSymbol[i];
                    string sym2 = indexToSymbol[j];
                    
                    vector<double> ret1 = priceHistory[sym1].getRecentReturns();
                    vector<double> ret2 = priceHistory[sym2].getRecentReturns();
                    
                    double corr = calculateCorrelation(ret1, ret2);
                    correlations[i][j] = corr;
                }
            }
        }
    }
    
    double calculateCorrelation(const vector<double>& x, const vector<double>& y) {
        if (x.size() != y.size() || x.size() < 2) return 0.0;
        
        double meanX = 0.0, meanY = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            meanX += x[i];
            meanY += y[i];
        }
        meanX /= x.size();
        meanY /= y.size();
        
        double numerator = 0.0, denomX = 0.0, denomY = 0.0;
        for (size_t i = 0; i < x.size(); i++) {
            double diffX = x[i] - meanX;
            double diffY = y[i] - meanY;
            numerator += diffX * diffY;
            denomX += diffX * diffX;
            denomY += diffY * diffY;
        }
        
        if (denomX == 0.0 || denomY == 0.0) return 0.0;
        return numerator / sqrt(denomX * denomY);
    }
    
    // Dijkstra's algorithm for optimal rebalancing path
    vector<int> dijkstraOptimalPath(const string& from, const string& to) {
        if (symbolToIndex.find(from) == symbolToIndex.end() || 
            symbolToIndex.find(to) == symbolToIndex.end()) {
            return vector<int>();
        }
        
        int src = symbolToIndex[from];
        int dest = symbolToIndex[to];
        
        vector<double> dist(numAssets, numeric_limits<double>::infinity());
        vector<int> parent(numAssets, -1);
        vector<bool> visited(numAssets, false);
        
        dist[src] = 0.0;
        
        priority_queue<pair<double, int>, vector<pair<double, int>>, greater<pair<double, int>>> pq;
        pq.push({0.0, src});
        
        while (!pq.empty()) {
            int u = pq.top().second;
            pq.pop();
            
            if (visited[u]) continue;
            visited[u] = true;
            
            for (int v = 0; v < numAssets; v++) {
                if (u != v && !visited[v]) {
                    double weight = transactionCosts[u][v];
                    if (dist[u] + weight < dist[v]) {
                        dist[v] = dist[u] + weight;
                        parent[v] = u;
                        pq.push({dist[v], v});
                    }
                }
            }
        }
        
        // Reconstruct path
        vector<int> path;
        int current = dest;
        while (current != -1) {
            path.push_back(current);
            current = parent[current];
        }
        reverse(path.begin(), path.end());
        
        return path.empty() || path[0] != src ? vector<int>() : path;
    }
    
    double getOptimalRebalancingCost(const string& from, const string& to) {
        vector<int> path = dijkstraOptimalPath(from, to);
        if (path.size() < 2) return numeric_limits<double>::infinity();
        
        double totalCost = 0.0;
        for (size_t i = 0; i < path.size() - 1; i++) {
            totalCost += transactionCosts[path[i]][path[i+1]];
        }
        return totalCost;
    }
    
    // Get symbols for external access
    vector<string> getAllSymbols() const {
        vector<string> symbols;
        for (const auto& pair : symbolToIndex) {
            symbols.push_back(pair.first);
        }
        return symbols;
    }
    
    double getTransactionCost(const string& from, const string& to) {
        if (symbolToIndex.find(from) == symbolToIndex.end() || 
            symbolToIndex.find(to) == symbolToIndex.end()) {
            return 0.001; // Default cost
        }
        int i = symbolToIndex[from];
        int j = symbolToIndex[to];
        return transactionCosts[i][j];
    }
    
    double getCorrelation(const string& sym1, const string& sym2) {
        if (symbolToIndex.find(sym1) == symbolToIndex.end() || 
            symbolToIndex.find(sym2) == symbolToIndex.end()) {
            return 0.0;
        }
        int i = symbolToIndex[sym1];
        int j = symbolToIndex[sym2];
        return correlations[i][j];
    }
    
    int getAssetIndex(const string& symbol) {
        auto it = symbolToIndex.find(symbol);
        return it != symbolToIndex.end() ? it->second : -1;
    }
    
    string getSymbolByIndex(int index) {
        auto it = indexToSymbol.find(index);
        return it != indexToSymbol.end() ? it->second : "";
    }
    
    int getNumAssets() const { return numAssets; }
};

// DSU Implementation for MST
struct UnionFind {
    int n, noofcomp, *parent, *rank;
    
    UnionFind(int a) {
        n = noofcomp = a;
        parent = new int[n + 1];
        rank = new int[n + 1];
        for (int i = 0; i <= n; i++) {
            parent[i] = i;
            rank[i] = 1;
        }
    }
    
    ~UnionFind() {
        delete[] parent;
        delete[] rank;
    }
    
    int find(int a) {
        if (parent[a] == a) return a;
        return parent[a] = find(parent[a]);
    }
    
    void unite(int a, int b) {
        int repa = find(a);
        int repb = find(b);
        if (repa == repb) return;
        
        if (rank[repa] >= rank[repb]) {
            parent[repb] = repa;
            rank[repa] += rank[repb];
        } else {
            parent[repa] = repb;
            rank[repb] += rank[repa];
        }
        noofcomp--;
    }
    
    int size() {
        return noofcomp;
    }
};

// MST-based Portfolio Optimizer
class PortfolioOptimizer {
private:
    AssetGraph& graph;
    
    struct Edge {
        int u, v;
        double weight;
        bool operator<(const Edge& other) const {
            return weight < other.weight;
        }
    };
    
public:
    PortfolioOptimizer(AssetGraph& g) : graph(g) {}
    
    // Construct MST using correlations (converted to distances)
    vector<Edge> constructMST() {
        vector<Edge> edges;
        int n = graph.getNumAssets();
        
        // Create edges from correlation matrix
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                string sym1 = graph.getSymbolByIndex(i);
                string sym2 = graph.getSymbolByIndex(j);
                double correlation = graph.getCorrelation(sym1, sym2);
                
                // Convert correlation to distance: d = sqrt(2(1-ρ))
                double distance = sqrt(2.0 * (1.0 - correlation));
                edges.push_back({i, j, distance});
            }
        }
        
        // Sort edges by weight (distance)
        sort(edges.begin(), edges.end());
        
        // Apply Kruskal's algorithm using UnionFind
        vector<Edge> mst;
        UnionFind uf(n);
        
        for (const Edge& e : edges) {
            if (uf.find(e.u) != uf.find(e.v)) {
                uf.unite(e.u, e.v);
                mst.push_back(e);
                if (mst.size() == n - 1) break;
            }
        }
        
        return mst;
    }
    
    // Calculate degree centrality from MST
    map<string, int> calculateDegreeCentrality() {
        vector<Edge> mst = constructMST();
        map<string, int> degreeCentrality;
        
        // Initialize all assets with degree 0
        for (const string& symbol : graph.getAllSymbols()) {
            degreeCentrality[symbol] = 0;
        }
        
        // Count degrees in MST
        for (const Edge& e : mst) {
            string sym1 = graph.getSymbolByIndex(e.u);
            string sym2 = graph.getSymbolByIndex(e.v);
            degreeCentrality[sym1]++;
            degreeCentrality[sym2]++;
        }
        
        return degreeCentrality;
    }
    
    // Inverse Degree Centrality Portfolio weights
    map<string, double> calculateIDCPWeights() {
        map<string, int> degreeCentrality = calculateDegreeCentrality();
        map<string, double> weights;
        double totalInverseWeight = 0.0;
        
        // Calculate inverse degree centrality weights
        for (const auto& pair : degreeCentrality) {
            string symbol = pair.first;
            int degree = max(1, pair.second); // Avoid division by zero
            
            // Get volatility for the symbol
            double volatility = 0.01; // Default volatility
            
            // IDCP weight: w = 1 / (volatility * degree_centrality)
            double weight = 1.0 / (volatility * degree);
            weights[symbol] = weight;
            totalInverseWeight += weight;
        }
        
        // Normalize weights to sum to 1
        for (auto& pair : weights) {
            pair.second /= totalInverseWeight;
        }
        
        return weights;
    }
    
    // Select peripheral assets (low degree centrality)
    vector<string> selectPeripheralAssets(int count = 10) {
        map<string, int> degreeCentrality = calculateDegreeCentrality();
        
        vector<pair<int, string>> assets;
        for (const auto& pair : degreeCentrality) {
            assets.push_back({pair.second, pair.first});
        }
        
        // Sort by degree (ascending - peripheral assets first)
        sort(assets.begin(), assets.end());
        
        vector<string> selected;
        for (int i = 0; i < min(count, (int)assets.size()); i++) {
            selected.push_back(assets[i].second);
        }
        
        return selected;
    }
};

// Enhanced Portfolio with graph-based rebalancing
class Portfolio {
private:
    string traderId;
    double cash;
    map<string, int> holdings;
    map<string, double> avgCost;
    map<string, double> targetWeights;
    AssetGraph& graph;

public:
    Portfolio(string id, double initialCash, AssetGraph& g) 
        : traderId(id), cash(initialCash), graph(g) {}

    bool canBuy(string symbol, double price, int quantity) const {
        return cash >= (price * quantity);
    }

    bool canSell(string symbol, int quantity) const {
        auto it = holdings.find(symbol); 
        return it != holdings.end() && it->second >= quantity;
    }

    void executeBuy(string symbol, double price, int quantity) {
        double cost = price * quantity;
        cash -= cost;
        
        int currentHolding = holdings[symbol];
        double currentAvgCost = avgCost[symbol];
        double totalCost = (currentHolding * currentAvgCost) + cost;
        int newHolding = currentHolding + quantity;
        
        holdings[symbol] = newHolding;
        avgCost[symbol] = totalCost / newHolding;
    }

    void executeSell(string symbol, double price, int quantity) {
        cash += (price * quantity);
        holdings[symbol] -= quantity;
        if (holdings[symbol] == 0) {
            holdings.erase(symbol);
            avgCost.erase(symbol);
        }
    }
    
    // Set target portfolio weights (from MST optimization)
    void setTargetWeights(const map<string, double>& weights) {
        targetWeights = weights;
    }
    
    // Calculate rebalancing trades using Dijkstra's algorithm
    vector<pair<string, pair<string, int>>> calculateOptimalRebalancing(const map<string, double>& currentPrices) {
        vector<pair<string, pair<string, int>>> trades; // {action, {symbol, quantity}}
        
        double totalValue = cash;
        for (const auto& holding : holdings) {
            if (currentPrices.find(holding.first) != currentPrices.end()) {
                totalValue += holding.second * currentPrices.at(holding.first);
            }
        }
        
        // Calculate current weights and target quantities
        map<string, double> currentWeights;
        map<string, int> targetQuantities;
        
        for (const auto& price : currentPrices) {
            string symbol = price.first;
            double currentPrice = price.second;
            
            int currentQty = getHolding(symbol);
            double currentValue = currentQty * currentPrice;
            currentWeights[symbol] = currentValue / totalValue;
            
            if (targetWeights.find(symbol) != targetWeights.end()) {
                double targetValue = totalValue * targetWeights.at(symbol);
                targetQuantities[symbol] = (int)(targetValue / currentPrice);
            }
        }
        
        // Find assets to sell and buy
        vector<pair<string, int>> toSell, toBuy;
        
        for (const auto& target : targetQuantities) {
            string symbol = target.first;
            int targetQty = target.second;
            int currentQty = getHolding(symbol);
            
            if (currentQty > targetQty) {
                toSell.push_back({symbol, currentQty - targetQty});
            } else if (currentQty < targetQty) {
                toBuy.push_back({symbol, targetQty - currentQty});
            }
        }
        
        // Use Dijkstra's algorithm to find optimal rebalancing paths
        for (const auto& sell : toSell) {
            for (const auto& buy : toBuy) {
                if (sell.second > 0 && buy.second > 0) {
                    vector<int> path = graph.dijkstraOptimalPath(sell.first, buy.first);
                    
                    if (!path.empty()) {
                        int quantity = min(sell.second, buy.second);
                        trades.push_back({"SELL", {sell.first, quantity}});
                        trades.push_back({"BUY", {buy.first, quantity}});
                        
                        // Update remaining quantities
                        const_cast<pair<string, int>&>(sell).second -= quantity;
                        const_cast<pair<string, int>&>(buy).second -= quantity;
                    }
                }
            }
        }
        
        return trades;
    }

    void displayPortfolio() const {
        cout << "\n=== PORTFOLIO: " << traderId << " ===\n";
        cout << "Cash: $" << fixed << setprecision(2) << cash << endl;
        cout << "Holdings:\n";
        for (const auto& holding : holdings) {
            cout << holding.first << ": " << holding.second << " shares @ $" 
                      << fixed << setprecision(2) << avgCost.at(holding.first) << endl;
        }
        
        if (!targetWeights.empty()) {
            cout << "\nTarget Weights:\n";
            for (const auto& weight : targetWeights) {
                cout << weight.first << ": " << fixed << setprecision(1) 
                          << weight.second * 100 << "%" << endl;
            }
        }
    }
    
    void displayOptimalRebalancing(const map<string, double>& currentPrices) {
        vector<pair<string, pair<string, int>>> trades = calculateOptimalRebalancing(currentPrices);
        
        cout << "\n=== OPTIMAL REBALANCING PLAN ===\n";
        for (const auto& trade : trades) {
            cout << trade.first << " " << trade.second.second 
                      << " shares of " << trade.second.first << endl;
        }
        
        cout << "\nTransaction Cost Analysis:\n";
        for (size_t i = 0; i < trades.size(); i += 2) {
            if (i + 1 < trades.size() && trades[i].first == "SELL" && trades[i+1].first == "BUY") {
                string fromSymbol = trades[i].second.first;
                string toSymbol = trades[i+1].second.first;
                double cost = graph.getOptimalRebalancingCost(fromSymbol, toSymbol);
                cout << fromSymbol << " -> " << toSymbol 
                          << ": Cost = " << fixed << setprecision(4) << cost << endl;
            }
        }
    }

    double getCash() const { return cash; }
    const map<string, int>& getHoldings() const { return holdings; }
    
    int getHolding(const string& symbol) const {
        auto it = holdings.find(symbol);
        return it != holdings.end() ? it->second : 0;
    }
};

// Main Trading Exchange with Graph-based Portfolio Optimization
class GraphTradingExchange {
private:
    map<string, map<double, queue<Order*>>> buyOrders;  // symbol -> price -> orders
    map<string, map<double, queue<Order*>>> sellOrders; // symbol -> price -> orders
    map<string, Portfolio*> portfolios;
    vector<Trade*> trades;
    AssetGraph assetGraph;
    PortfolioOptimizer optimizer;
    map<string, double> currentPrices;

public:
    GraphTradingExchange() : optimizer(assetGraph) {
        // Initialize with some sample assets
        vector<string> initialAssets = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"};
        for (const string& symbol : initialAssets) {
            assetGraph.addAsset(symbol);
            currentPrices[symbol] = 100.0 + (rand() % 50); // Random initial prices
        }
    }
    
    ~GraphTradingExchange() {
        for (auto& portfolio : portfolios) {
            delete portfolio.second;
        }
        for (Trade* trade : trades) {
            delete trade;
        }
    }
    
    void registerTrader(const string& traderId, double initialCash) {
        if (portfolios.find(traderId) == portfolios.end()) {
            portfolios[traderId] = new Portfolio(traderId, initialCash, assetGraph);
            cout << "Trader " << traderId << " registered with $" << initialCash << endl;
        }
    }
    
    void updateMarketPrice(const string& symbol, double price) {
        currentPrices[symbol] = price;
        assetGraph.updatePrice(symbol, price);
    }
    
    void placeOrder(Order* order) {
        cout << "\n--- PLACING ORDER ---\n";
        cout << "Order ID: " << order->id << ", Symbol: " << order->symbol 
                  << ", Side: " << (order->side == OrderSide::BUY ? "BUY" : "SELL")
                  << ", Quantity: " << order->quantity << ", Price: $" << order->price << endl;
        
        if (order->side == OrderSide::BUY) {
            processBuyOrder(order);
        } else {
            processSellOrder(order);
        }
    }
    
    void processBuyOrder(Order* buyOrder) {
        auto& sellBook = sellOrders[buyOrder->symbol];
        
        for (auto it = sellBook.begin(); it != sellBook.end() && 
             !buyOrder->isFullyFilled(); ++it) {
            
            if (it->first <= buyOrder->price) {
                auto& sellQueue = it->second;
                
                while (!sellQueue.empty() && !buyOrder->isFullyFilled()) {
                    Order* sellOrder = sellQueue.front();
                    
                    int tradeQuantity = min(buyOrder->getRemainingQuantity(), 
                                          sellOrder->getRemainingQuantity());
                    double tradePrice = sellOrder->price;
                    
                    executeTrade(buyOrder, sellOrder, tradeQuantity, tradePrice);
                    
                    if (sellOrder->isFullyFilled()) {
                        sellQueue.pop();
                        sellOrder->status = OrderStatus::FILLED;
                    }
                }
                
                if (sellQueue.empty()) {
                    sellBook.erase(it--);
                }
            }
        }
        
        if (!buyOrder->isFullyFilled()) {
            buyOrders[buyOrder->symbol][buyOrder->price].push(buyOrder);
            buyOrder->status = buyOrder->filledQuantity > 0 ? 
                             OrderStatus::PARTIALLY_FILLED : OrderStatus::PENDING;
        } else {
            buyOrder->status = OrderStatus::FILLED;
        }
    }
    
    void processSellOrder(Order* sellOrder) {
        auto& buyBook = buyOrders[sellOrder->symbol];
        
        for (auto it = buyBook.rbegin(); it != buyBook.rend() && 
             !sellOrder->isFullyFilled(); ++it) {
            
            if (it->first >= sellOrder->price) {
                auto& buyQueue = it->second;
                
                while (!buyQueue.empty() && !sellOrder->isFullyFilled()) {
                    Order* buyOrder = buyQueue.front();
                    
                    int tradeQuantity = min(sellOrder->getRemainingQuantity(), 
                                          buyOrder->getRemainingQuantity());
                    double tradePrice = buyOrder->price;
                    
                    executeTrade(buyOrder, sellOrder, tradeQuantity, tradePrice);
                    
                    if (buyOrder->isFullyFilled()) {
                        buyQueue.pop();
                        buyOrder->status = OrderStatus::FILLED;
                    }
                }
                
                if (buyQueue.empty()) {
                    buyBook.erase(next(it).base());
                }
            }
        }
        
        if (!sellOrder->isFullyFilled()) {
            sellOrders[sellOrder->symbol][sellOrder->price].push(sellOrder);
            sellOrder->status = sellOrder->filledQuantity > 0 ? 
                              OrderStatus::PARTIALLY_FILLED : OrderStatus::PENDING;
        } else {
            sellOrder->status = OrderStatus::FILLED;
        }
    }
    
    void executeTrade(Order* buyOrder, Order* sellOrder, int quantity, double price) {
        Trade* trade = new Trade(buyOrder->symbol, price, quantity, 
                               buyOrder->trader, sellOrder->trader);
        trades.push_back(trade);
        
        buyOrder->filledQuantity += quantity;
        sellOrder->filledQuantity += quantity;
        
        // Update portfolios
        if (portfolios.find(buyOrder->trader) != portfolios.end()) {
            portfolios[buyOrder->trader]->executeBuy(buyOrder->symbol, price, quantity);
        }
        
        if (portfolios.find(sellOrder->trader) != portfolios.end()) {
            portfolios[sellOrder->trader]->executeSell(sellOrder->symbol, price, quantity);
        }
        
        cout << "TRADE EXECUTED: " << quantity << " shares of " << buyOrder->symbol 
                  << " at $" << price << " (Buyer: " << buyOrder->trader 
                  << ", Seller: " << sellOrder->trader << ")" << endl;
        
        // Update market price
        updateMarketPrice(buyOrder->symbol, price);
    }
    
    void displayOrderBook(const string& symbol) {
        cout << "\n=== ORDER BOOK: " << symbol << " ===\n";
        
        cout << "SELL ORDERS (Ask):\n";
        auto& sellBook = sellOrders[symbol];
        for (auto it = sellBook.rbegin(); it != sellBook.rend(); ++it) {
            int totalQuantity = 0;
            queue<Order*> tempQueue = it->second;
            while (!tempQueue.empty()) {
                totalQuantity += tempQueue.front()->getRemainingQuantity();
                tempQueue.pop();
            }
            if (totalQuantity > 0) {
                cout << "  $" << it->first << " x " << totalQuantity << endl;
            }
        }
        
        cout << "BUY ORDERS (Bid):\n";
        auto& buyBook = buyOrders[symbol];
        for (auto it = buyBook.rbegin(); it != buyBook.rend(); ++it) {
            int totalQuantity = 0;
            queue<Order*> tempQueue = it->second;
            while (!tempQueue.empty()) {
                totalQuantity += tempQueue.front()->getRemainingQuantity();
                tempQueue.pop();
            }
            if (totalQuantity > 0) {
                cout << "  $" << it->first << " x " << totalQuantity << endl;
            }
        }
    }
    
    void optimizePortfolio(const string& traderId) {
        if (portfolios.find(traderId) == portfolios.end()) {
            cout << "Trader not found!" << endl;
            return;
        }
        
        cout << "\n=== PORTFOLIO OPTIMIZATION FOR " << traderId << " ===\n";
        
        // Calculate IDCP weights using MST
        map<string, double> idcpWeights = optimizer.calculateIDCPWeights();
        
        cout << "MST-based IDCP Weights:\n";
        for (const auto& weight : idcpWeights) {
            cout << weight.first << ": " << fixed << setprecision(1) 
                      << weight.second * 100 << "%" << endl;
        }
        
        // Set target weights for the portfolio
        portfolios[traderId]->setTargetWeights(idcpWeights);
        
        // Display optimal rebalancing plan
        portfolios[traderId]->displayOptimalRebalancing(currentPrices);
    }
    
    void displayMSTAnalysis() {
        cout << "\n=== MINIMUM SPANNING TREE ANALYSIS ===\n";
        
        // Show degree centrality
        map<string, int> degreeCentrality = optimizer.calculateDegreeCentrality();
        cout << "Degree Centrality (MST):\n";
        for (const auto& dc : degreeCentrality) {
            cout << dc.first << ": " << dc.second << endl;
        }
        
        // Show peripheral assets
        vector<string> peripheral = optimizer.selectPeripheralAssets(3);
        cout << "\nMost Peripheral Assets (Low Risk):\n";
        for (const string& symbol : peripheral) {
            cout << "- " << symbol << endl;
        }
        
        // Show transaction cost matrix (sample)
        cout << "\nTransaction Cost Matrix (Sample):\n";
        vector<string> symbols = assetGraph.getAllSymbols();
        cout << setw(8) << "";
        for (size_t i = 0; i < min(size_t(3), symbols.size()); i++) {
            cout << setw(8) << symbols[i];
        }
        cout << endl;
        
        for (size_t i = 0; i < min(size_t(3), symbols.size()); i++) {
            cout << setw(8) << symbols[i];
            for (size_t j = 0; j < min(size_t(3), symbols.size()); j++) {
                double cost = assetGraph.getTransactionCost(symbols[i], symbols[j]);
                cout << setw(8) << fixed << setprecision(4) << cost;
            }
            cout << endl;
        }
    }
    
    void displayPortfolio(const string& traderId) {
        if (portfolios.find(traderId) != portfolios.end()) {
            portfolios[traderId]->displayPortfolio();
        }
    }
    
    void displayTradeHistory() {
        cout << "\n=== TRADE HISTORY ===\n";
        for (const Trade* trade : trades) {
            cout << "Trade " << trade->id << ": " << trade->quantity 
                      << " shares of " << trade->symbol << " at $" << trade->price
                      << " (Buyer: " << trade->buyer << ", Seller: " << trade->seller << ")" << endl;
        }
    }
    
    void simulateMarketData() {
        cout << "\n=== SIMULATING MARKET DATA ===\n";
        
        // Simulate price movements for graph construction
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dis(0.0, 0.02); // 2% daily volatility
        
        vector<string> symbols = assetGraph.getAllSymbols();
        for (int day = 0; day < 30; day++) {
            for (const string& symbol : symbols) {
                double currentPrice = currentPrices[symbol];
                double change = dis(gen);
                double newPrice = currentPrice * (1.0 + change);
                updateMarketPrice(symbol, newPrice);
            }
        }
        
        cout << "Generated 30 days of price data for " << symbols.size() << " assets\n";
        cout << "Current Prices:\n";
        for (const auto& price : currentPrices) {
            cout << price.first << ": $" << fixed << setprecision(2) << price.second << endl;
        }
    }
};

// Demo function implementing the research paper concepts
void demonstrateGraphBasedTrading() {
    cout << "=== GRAPH-BASED TRADING EXCHANGE DEMO ===\n";
    cout << "Implementing Research Paper: 'Dynamic Portfolio Rebalancing using GNNs and Dijkstra'\n";
    cout << "and 'Portfolio Optimization Using Minimum Spanning Tree Model'\n\n";
    
    GraphTradingExchange exchange;
    
    // Register traders
    exchange.registerTrader("Alice", 50000.0);
    exchange.registerTrader("Bob", 75000.0);
    
    // Simulate market data for graph construction
    exchange.simulateMarketData();
    
    // Display MST analysis
    exchange.displayMSTAnalysis();
    
    // Place some initial orders
    exchange.placeOrder(new Order("AAPL", OrderSide::BUY, OrderType::LIMIT, 150.0, 100, "Alice"));
    exchange.placeOrder(new Order("MSFT", OrderSide::BUY, OrderType::LIMIT, 200.0, 50, "Alice"));
    exchange.placeOrder(new Order("AAPL", OrderSide::SELL, OrderType::LIMIT, 152.0, 50, "Bob"));
    exchange.placeOrder(new Order("GOOGL", OrderSide::BUY, OrderType::LIMIT, 2500.0, 20, "Bob"));
    
    // Display portfolios
    exchange.displayPortfolio("Alice");
    exchange.displayPortfolio("Bob");
    
    // Optimize Alice's portfolio using MST-based approach
    exchange.optimizePortfolio("Alice");
    
    // Display order books
    exchange.displayOrderBook("AAPL");
    
    // Display trade history
    exchange.displayTradeHistory();
    
    cout << "\n=== DEMONSTRATION COMPLETE ===\n";
    cout << "The system successfully implements:\n";
    cout << "1. Graph Neural Network concepts for transaction cost prediction\n";
    cout << "2. Dijkstra's algorithm for optimal rebalancing paths\n";
    cout << "3. Minimum Spanning Tree for portfolio structure analysis\n";
    cout << "4. Inverse Degree Centrality Portfolio (IDCP) allocation\n";
    cout << "5. Dynamic portfolio rebalancing with cost optimization\n";
}

int main() {
    srand(time(nullptr));
    demonstrateGraphBasedTrading();
    return 0;
}